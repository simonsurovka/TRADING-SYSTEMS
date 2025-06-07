// Imports and Dependencies
use eframe::egui::{self, Color32, Shadow, Button, Ui, RichText, Vec2, Visuals, DragValue, TextEdit, ComboBox, ScrollArea, CentralPanel, Context};
use chrono::{DateTime, Utc};
use ibkr_rust::{TwsConnection, Contract, Order as IbkrOrder};
use std::time::{Duration, Instant};
use log::{info, warn, error, debug, trace};
use std::sync::{Arc, Mutex};
use std::thread;
use std::fs::{OpenOptions, File, rename, remove_file, metadata};
use std::io::{Write, Seek, SeekFrom};
use std::net::IpAddr;
use std::path::Path;
use std::env;
use openssl::symm::{Cipher, Crypter, Mode};
use tokio::sync::Mutex as AsyncMutex;
use tokio::runtime::Runtime;

// Modularization: Each major feature is a module for scalability and clarity
pub mod user;         // User management
pub mod order;        // Order management
pub mod market_data;  // Market data
pub mod risk;         // Risk management
pub mod logging;      // Logging and audit trail
pub mod settings;     // Persistent settings
pub mod backtest;     // Backtesting engine
pub mod api;          // API integration
pub mod ui;           // UI helpers/components
pub mod tests;        // Test utilities

// Re-export module contents for easier access throughout the codebase
pub use user::*;
pub use order::*;
pub use market_data::*;
pub use risk::*;
pub use logging::*;
pub use settings::*;
pub use backtest::*;
pub use api::*;
pub use ui::*;

// User Role Enum
#[derive(Clone, Debug, PartialEq)]
pub enum UserRole {
    Admin,      // Full access to all features and settings
    Trader,     // Can trade, but cannot change system settings
    Observer,   // Read-only access, cannot trade
}

// User Session Struct
#[derive(Clone, Debug)]
pub struct UserSession {
    pub username: String,
    pub role: UserRole,
    pub authenticated: bool,
    pub last_active: Instant,
    pub two_factor_passed: bool,
    pub ip_address: Option<IpAddr>,
    pub encrypted_password: Option<Vec<u8>>,
    pub session_warning_shown: bool,
    pub session_warning_time: Option<Instant>,
}

// Implementation of UserSession methods
impl UserSession {
    pub fn new(username: &str, role: UserRole) -> Self {
        Self {
            username: username.to_string(),
            role,
            authenticated: false,
            last_active: Instant::now(),
            two_factor_passed: false,
            ip_address: None,
            encrypted_password: None,
            session_warning_shown: false,
            session_warning_time: None,
        }
    }
    pub fn is_active(&self, timeout_secs: u64) -> bool {
        self.last_active.elapsed() < Duration::from_secs(timeout_secs)
    }
    pub fn time_left(&self, timeout_secs: u64) -> Option<u64> {
        let elapsed = self.last_active.elapsed();
        if elapsed < Duration::from_secs(timeout_secs) {
            Some(timeout_secs - elapsed.as_secs())
        } else {
            None
        }
    }
    pub fn should_warn(&mut self, timeout_secs: u64, warn_before_secs: u64) -> bool {
        let elapsed = self.last_active.elapsed();
        let time_left = timeout_secs.saturating_sub(elapsed.as_secs());
        if time_left <= warn_before_secs && !self.session_warning_shown {
            self.session_warning_shown = true;
            self.session_warning_time = Some(Instant::now());
            true
        } else {
            false
        }
    }
    pub fn extend_session(&mut self) {
        self.last_active = Instant::now();
        self.session_warning_shown = false;
        self.session_warning_time = None;
    }
}

// Error Handling and Validation Structures
#[derive(Debug, Clone)]
pub enum FieldError {
    Required,
    InvalidFormat,
    BelowMinimum { min: Decimal },
    AboveMaximum { max: Decimal },
    NotTickAligned { tick: Decimal },
    PriceOutOfRange { min: Decimal, max: Decimal },
    InsufficientFunds,
    Unauthorized,
    OrderTypeNotSupported,
    SessionExpired,
    ApiRateLimited,
    Other(String),
}

// Implement Display for FieldError for user-friendly error messages.
impl std::fmt::Display for FieldError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FieldError::Required => write!(f, "This field is required"),
            FieldError::InvalidFormat => write!(f, "Invalid format"),
            FieldError::BelowMinimum { min } => write!(f, "Must be at least {}", min),
            FieldError::AboveMaximum { max } => write!(f, "Must not exceed {}", max),
            FieldError::NotTickAligned { tick } => write!(f, "Price must be a multiple of tick size {}", tick),
            FieldError::PriceOutOfRange { min, max } => write!(f, "Price must be between {} and {}", min, max),
            FieldError::InsufficientFunds => write!(f, "Insufficient funds or margin for this order"),
            FieldError::Unauthorized => write!(f, "You are not authorized to perform this action"),
            FieldError::OrderTypeNotSupported => write!(f, "This order type is not supported"),
            FieldError::SessionExpired => write!(f, "Session expired due to inactivity"),
            FieldError::ApiRateLimited => write!(f, "API rate limit reached, please wait and try again"),
            FieldError::Other(msg) => write!(f, "{}", msg),
        }
    }
}

// Struct to collect validation errors for multiple fields.
#[derive(Debug, Default, Clone)]
pub struct ValidationErrors {
    pub errors: HashMap<String, Vec<FieldError>>, 
}

// Implementation of ValidationErrors methods
impl ValidationErrors {
    // Add an error for a specific field
    pub fn add(&mut self, field: &str, err: FieldError) {
        self.errors.entry(field.to_string()).or_default().push(err);
    }
    // Clear all errors
    pub fn clear(&mut self) {
        self.errors.clear();
    }
    // Get errors for a specific field
    pub fn get(&self, field: &str) -> Option<&Vec<FieldError>> {
        self.errors.get(field)
    }
    // Check if there are no errors
    pub fn is_empty(&self) -> bool {
        self.errors.is_empty()
    }
    // Get a summary of all errors as strings
    pub fn summary(&self) -> Vec<String> {
        self.errors
            .iter()
            .flat_map(|(f, errs)| errs.iter().map(move |e| format!("{}: {}", f, e)))
            .collect()
    }
}

// IBKR Connection Configuration
#[derive(Clone)]
pub struct IbkrConfig {
    pub host: String,
    pub port: u16,
    pub timeout_secs: u64,
    pub account_id: String,
    pub margin: Decimal,
    pub cash_balance: Decimal,
    pub encrypted_api_key: Option<Vec<u8>>,
    pub api_rate_limit: u32,
}

// Default implementation for IbkrConfig
impl Default for IbkrConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 7497,
            timeout_secs: 5,
            account_id: String::new(),
            margin: Decimal::new(100_000, 0),
            cash_balance: Decimal::new(100_000, 0),
            encrypted_api_key: None,
            api_rate_limit: 40,
        }
    }
}

// Market Data WebSocket Integration
pub struct MarketDataWebSocket {
    pub connected: bool,
    pub last_error: Option<String>,
    pub current_symbol: Option<String>,
    pub reconnect_attempts: u32,
    pub max_reconnect_attempts: u32,
    pub use_tls: bool,
    pub last_message_time: Option<Instant>,
}

// Implementation of MarketDataWebSocket methods
impl MarketDataWebSocket {
    pub fn new() -> Self {
        Self {
            connected: false,
            last_error: None,
            current_symbol: None,
            reconnect_attempts: 0,
            max_reconnect_attempts: 5,
            use_tls: true,
            last_message_time: None,
        }
    }
    pub async fn connect_async(&mut self, _host: &str, _port: u16) -> Result<(), String> {
        self.connected = true;
        self.reconnect_attempts = 0;
        self.last_error = None;
        self.last_message_time = Some(Instant::now());
        Ok(())
    }
    pub fn subscribe(&mut self, symbol: &str) {
        if self.connected {
            self.current_symbol = Some(symbol.to_string());
            self.last_message_time = Some(Instant::now());
        }
    }
    pub fn unsubscribe(&mut self, symbol: &str) {
        if self.connected && self.current_symbol.as_deref() == Some(symbol) {
            self.current_symbol = None;
        }
    }
    pub fn disconnect(&mut self) {
        self.connected = false;
        self.current_symbol = None;
    }
    pub fn handle_disconnect(&mut self) {
        self.connected = false;
        self.last_error = Some("WebSocket disconnected".to_string());
        self.reconnect_attempts += 1;
        if self.reconnect_attempts <= self.max_reconnect_attempts {
            let wait = 2u64.pow(self.reconnect_attempts) * 100;
            std::thread::sleep(Duration::from_millis(wait));
        }
    }
}

// Audit Trail and Logging
pub struct AuditTrail {
    file: Mutex<File>,
    path: String,
    max_size_bytes: u64,
}

// Implementation of AuditTrail methods
impl AuditTrail {
    pub fn new(path: &str) -> Self {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .unwrap();
        Self {
            file: Mutex::new(file),
            path: path.to_string(),
            max_size_bytes: 5 * 1024 * 1024, // 5 MB
        }
    }
    pub fn log(&self, entry: &str) {
        let mut file = self.file.lock().unwrap();
        let _ = writeln!(file, "{}", entry);
        let _ = file.flush();
        drop(file);
        self.rotate_if_needed();
    }
    fn rotate_if_needed(&self) {
        if let Ok(meta) = metadata(&self.path) {
            if meta.len() > self.max_size_bytes {
                let rotated = format!("{}.{}", &self.path, Utc::now().format("%Y%m%d%H%M%S"));
                let _ = self.file.lock().unwrap().flush();
                let _ = rename(&self.path, &rotated);
                let _ = OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&self.path);
            }
        }
    }
}

// Persistent Settings 
#[derive(Default, Clone, serde::Serialize, serde::Deserialize)]
pub struct PersistentSettings {
    pub theme: String,
    pub dom_levels: usize,
    pub last_symbol: String,
    pub last_account: usize,
    pub simulated_mode: bool,
    pub custom_theme: Option<String>,
}

// Implementation of PersistentSettings methods
impl PersistentSettings {
    pub fn load_from_file(path: &str) -> Self {
        if let Ok(data) = std::fs::read_to_string(path) {
            if let Ok(settings) = serde_json::from_str(&data) {
                return settings;
            }
        }
        Self::default()
    }
    pub fn save_to_file(&self, path: &str) {
        if let Ok(json) = serde_json::to_string_pretty(self) {
            let _ = std::fs::write(path, json);
        }
    }
}

// Main Application Structure 
#[derive(Clone, Debug)]
pub enum OrderStatus {
    Pending,
    Filled,
    PartiallyFilled(Decimal),
    Cancelled,
    Rejected(String),
    FOKFailed,
    IOCPartial(Decimal),
}

pub struct TradingApp {
    order_book: OrderBook,
    symbol: String,
    quantity: String,
    price: String,
    status_message: String,
    tws_connection: Option<TwsConnection>,
    connection_status: ConnectionStatus,
    error_fields: ValidationErrors,
    show_status: bool,
    message_history: Vec<(DateTime<Utc>, String)>,
    sort_by_price: bool,
    ibkr_config: IbkrConfig,
    debug_mode: bool,
    dom_levels: usize,
    user_session: UserSession,
    websocket: Option<Arc<AsyncMutex<MarketDataWebSocket>>>,
    audit_trail: Arc<AuditTrail>,
    order_type: String,
    order_duration: String,
    max_order_size: Decimal,
    min_tick_size: Decimal,
    min_price: Decimal,
    max_price: Decimal,
    margin_available: Decimal,
    persistent_settings: PersistentSettings,
    accounts: Vec<IbkrConfig>,
    selected_account: usize,
    dark_mode: bool,
    last_symbol_subscribed: Option<String>,
    session_timeout_secs: u64,
    session_warn_before_secs: u64,
    order_history: Vec<OrderHistoryEntry>,
    custom_order_params: CustomOrderParams,
    simulated_mode: bool,
    order_statuses: HashMap<u64, OrderStatus>,
    drag_order: Option<(bool, Decimal, Decimal)>,
    price_history: Vec<(DateTime<Utc>, Decimal)>,
    backtest_results: Option<BacktestResults>,
    last_api_call: Option<Instant>,
    api_calls_this_minute: u32,
    last_api_minute: Option<u64>,
    risk_alerts: Vec<String>,
}

/// Struct for a single order history entry 
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct OrderHistoryEntry {
    pub timestamp: DateTime<Utc>,      
    pub user: String,                  
    pub action: String,                
    pub symbol: String,                
    pub size: Decimal,                 
    pub price: Decimal,                
    pub order_type: String,            
    pub duration: String,              
    pub result: String,                
    pub account_id: String,            
}

// Struct for custom order parameters (iceberg, trailing, etc.)
#[derive(Clone, Debug, Default)]
pub struct CustomOrderParams {
    pub iceberg_size: Option<Decimal>,     
    pub limit_offset: Option<Decimal>,     
    pub simulated: bool,                   
    pub trailing_amount: Option<Decimal>,  
    pub fok: bool,                         
    pub ioc: bool,                         
}

// Struct for backtest results (PnL, drawdown, trades, etc.)
pub struct BacktestResults {
    pub trades: Vec<OrderHistoryEntry>,    
    pub pnl: Decimal,                      
    pub max_drawdown: Decimal,             
    pub sharpe: f64,                       
}

// Implementation of TradingApp methods
impl TradingApp {
    /// Create a new TradingApp instance (called at startup).
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let audit_trail = Arc::new(AuditTrail::new("audit_trail.log"));
        let persistent_settings = PersistentSettings::load_from_file("settings.json");
        let accounts = vec![IbkrConfig::default()];
        let dom_levels = if persistent_settings.dom_levels == 0 {
            10
        } else {
            persistent_settings.dom_levels
        };
        Self {
            order_book: OrderBook::default(),
            symbol: persistent_settings.last_symbol.clone(),
            quantity: String::new(),
            price: String::new(),
            status_message: String::new(),
            tws_connection: None,
            connection_status: ConnectionStatus::Disconnected,
            error_fields: ValidationErrors::default(),
            show_status: true,
            message_history: Vec::new(),
            sort_by_price: true,
            ibkr_config: accounts[0].clone(),
            debug_mode: false,
            dom_levels,
            user_session: UserSession::new("guest", UserRole::Observer),
            websocket: None,
            audit_trail,
            order_type: "Limit".to_string(),
            order_duration: "Day".to_string(),
            max_order_size: Decimal::new(10_000, 0),
            min_tick_size: Decimal::new(5, 2),
            min_price: Decimal::new(1, 2),
            max_price: Decimal::new(1_000_000, 0),
            margin_available: Decimal::new(100_000, 0),
            persistent_settings: persistent_settings.clone(),
            accounts,
            selected_account: 0,
            dark_mode: persistent_settings.theme == "dark",
            last_symbol_subscribed: None,
            session_timeout_secs: 15 * 60,
            session_warn_before_secs: 60,
            order_history: Vec::new(),
            custom_order_params: CustomOrderParams::default(),
            simulated_mode: persistent_settings.simulated_mode,
            order_statuses: HashMap::new(),
            drag_order: None,
            price_history: Vec::new(),
            backtest_results: None,
            last_api_call: None,
            api_calls_this_minute: 0,
            last_api_minute: None,
            risk_alerts: Vec::new(),
        }
    }

}

// OrderBook

use std::collections::{BinaryHeap, HashSet};
use std::cmp::Reverse;

// Represents a price/size order entry.
#[derive(Clone, Debug, PartialEq)]
pub struct OrderEntry {
    pub price: Decimal,
    pub size: Decimal,
    pub timestamp: Option<DateTime<Utc>>,
}

impl OrderEntry {
    pub fn new(price: Decimal, size: Decimal, timestamp: Option<DateTime<Utc>>) -> Self {
        Self { price, size, timestamp }
    }
}

// OrderBook structure 
#[derive(Default, Debug)]
pub struct OrderBook {
    pub bids: BinaryHeap<(Decimal, Decimal)>, 
    pub asks: BinaryHeap<(Reverse<Decimal>, Decimal)>, 
    pub total_bid_orders: usize,
    pub total_ask_orders: usize,
}

impl OrderBook {
    pub fn new() -> Self {
        OrderBook {
            bids: BinaryHeap::new(),
            asks: BinaryHeap::new(),
            total_bid_orders: 0,
            total_ask_orders: 0,
        }
    }

    // Add an order to the book.
    pub fn add_order(&mut self, price: Decimal, size: Decimal, is_bid: bool) {
        if is_bid {
            self.bids.push((price, size));
            self.total_bid_orders += 1;
        } else {
            self.asks.push((Reverse(price), size));
            self.total_ask_orders += 1;
        }
    }

    // Remove a single order matching price and size from the book.
    pub fn remove_order(&mut self, price: Decimal, size: Decimal, is_bid: bool) -> bool {
        let mut removed = false;
        if is_bid {
            let mut new_bids = BinaryHeap::new();
            while let Some((p, s)) = self.bids.pop() {
                if !removed && p == price && s == size {
                    removed = true;
                    self.total_bid_orders = self.total_bid_orders.saturating_sub(1);
                    continue;
                }
                new_bids.push((p, s));
            }
            self.bids = new_bids;
        } else {
            let mut new_asks = BinaryHeap::new();
            while let Some((rp, s)) = self.asks.pop() {
                if !removed && rp.0 == price && s == size {
                    removed = true;
                    self.total_ask_orders = self.total_ask_orders.saturating_sub(1);
                    continue;
                }
                new_asks.push((rp, s));
            }
            self.asks = new_asks;
        }
        removed
    }

    pub fn clear(&mut self) {
        self.bids.clear();
        self.asks.clear();
        self.total_bid_orders = 0;
        self.total_ask_orders = 0;
    }


    pub fn get_best_bid(&self) -> Option<Decimal> {
        self.bids.peek().map(|(p, _)| *p)
    }
    pub fn get_best_ask(&self) -> Option<Decimal> {
        self.asks.peek().map(|(rp, _)| rp.0)
    }
  
    pub fn get_total_volumes(&self) -> (Decimal, Decimal) {
        let bid_volume: Decimal = self.bids.iter().map(|(_, size)| *size).sum();
        let ask_volume: Decimal = self.asks.iter().map(|(_, size)| *size).sum();
        (bid_volume, ask_volume)
    }

    pub fn get_depth(&self) -> (usize, usize) {
        let bid_levels = self.bids.iter().map(|(price, _)| *price).collect::<HashSet<_>>().len();
        let ask_levels = self.asks.iter().map(|(reverse_price, _)| reverse_price.0).collect::<HashSet<_>>().len();
        (bid_levels, ask_levels)
    }

    pub fn get_average_bid_price(&self) -> Option<Decimal> {
        let total_size: Decimal = self.bids.iter().map(|(_, size)| *size).sum();
        if total_size.is_zero() {
            None
        } else {
            let weighted_sum: Decimal = self.bids.iter().map(|(price, size)| *price * *size).sum();
            Some(weighted_sum / total_size)
        }
    }

    pub fn get_average_ask_price(&self) -> Option<Decimal> {
        let total_size: Decimal = self.asks.iter().map(|(_, size)| *size).sum();
        if total_size.is_zero() {
            None
        } else {
            let weighted_sum: Decimal = self.asks.iter().map(|(reverse_price, size)| reverse_price.0 * *size).sum();
            Some(weighted_sum / total_size)
        }
    }

    // Update the order book 
    pub fn update_from_market_feed(&mut self, market_data: &ibkr_rust::MarketData) {
        // Clear current state
        self.bids.clear();
        self.asks.clear();
        self.total_bid_orders = 0;
        self.total_ask_orders = 0;

        // Efficiently update bids
        if let Some(bids) = &market_data.bids {
            self.total_bid_orders = bids.len();
            for &(price, size) in bids {
                if size > Decimal::ZERO {
                    self.bids.insert(price, size);
                }
            }
        }

        // Efficiently update asks
        if let Some(asks) = &market_data.asks {
            self.total_ask_orders = asks.len();
            for &(price, size) in asks {
                if size > Decimal::ZERO {
                    self.asks.insert(Reverse(price), size);
                }
            }
        }
    }

    // Returns a sorted vector of all bid levels 
    pub fn get_sorted_bids(&self) -> Vec<(Decimal, Decimal)> {
        let mut bids: Vec<_> = self.bids.iter().map(|(p, s)| (*p, *s)).collect();
        bids.sort_unstable_by(|a, b| b.0.cmp(&a.0));
        bids
    }

    // Returns a sorted vector of all ask levels 
    pub fn get_sorted_asks(&self) -> Vec<(Decimal, Decimal)> {
        let mut asks: Vec<_> = self.asks.iter().map(|(rp, s)| (rp.0, *s)).collect();
        asks.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        asks
    }
}

// Connection Status Enum 
#[derive(PartialEq, Debug, Clone, Copy)]
pub enum ConnectionStatus {
    Connected,
    Connecting,
    Disconnected,
}

// Order struct 
#[derive(Clone, Debug, PartialEq)]
pub struct Order {
    pub price: Decimal,
    pub size: Decimal,
    pub timestamp: DateTime<Utc>,
}

impl Order {
    pub fn new(price: Decimal, size: Decimal, timestamp: DateTime<Utc>) -> Self {
        Self { price, size, timestamp }
    }
}

// eframe::App Implementation 

impl eframe::App for TradingApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Theme Customization 
        if self.dark_mode {
            ctx.set_visuals(Visuals::dark());
        } else {
            ctx.set_visuals(Visuals::light());
        }
        // Session Expiry and Warning 
        let mut show_session_warning = false;
        if self.user_session.authenticated {
            if !self.user_session.is_active(self.session_timeout_secs) {
                self.user_session.authenticated = false;
                self.status_message = "Session expired due to inactivity. Please log in again.".to_string();
            } else if self.user_session.should_warn(self.session_timeout_secs, self.session_warn_before_secs) {
                show_session_warning = true;
            }
        }

        CentralPanel::default().show(ctx, |ui| {
            // Session Timeout Warning Modal 
            if show_session_warning {
                egui::Window::new("Session Timeout Warning")
                    .collapsible(false)
                    .resizable(false)
                    .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                    .show(ctx, |ui| {
                        ui.label(RichText::new("Your session will expire soon due to inactivity.").color(Color32::YELLOW));
                        if let Some(time_left) = self.user_session.time_left(self.session_timeout_secs) {
                            ui.label(format!("Time left: {} seconds", time_left));
                        }
                        if ui.button("Extend Session").clicked() {
                            self.user_session.extend_session();
                        }
                    });
            }

            // User Authentication and Session Management 
            if !self.user_session.authenticated {
                ui.group(|ui| {
                    ui.heading("User Login");
                    let mut username = self.user_session.username.clone();
                    let mut password = String::new();
                    let mut two_factor = String::new();
                    let mut login_error = String::new();
                    ui.horizontal(|ui| {
                        ui.label("Username:");
                        ui.text_edit_singleline(&mut username)
                            .on_hover_text("Enter your username (admin, trader, observer)");
                    });
                    ui.horizontal(|ui| {
                        ui.label("Password:");
                        ui.add(TextEdit::singleline(&mut password).password(true))
                            .on_hover_text("Enter your password");
                    });
                    ui.horizontal(|ui| {
                        ui.label("2FA Code:");
                        ui.text_edit_singleline(&mut two_factor)
                            .on_hover_text("Enter your 2FA code from Google Authenticator or similar app");
                    });
                    if ui.button("Login").clicked() {
                        match self.authenticate_user(&username, &password, Some(&two_factor)) {
                            Ok(role) => {
                                self.status_message = format!("Login successful as {:?}", role);
                                self.error_fields.clear();
                            }
                            Err(e) => {
                                self.status_message = format!("Login failed: {}", e);
                                login_error = e;
                                self.error_fields.add("login", FieldError::Other(login_error.clone()));
                            }
                        }
                    }
                    if let Some(errors) = self.error_fields.get("login") {
                        for err in errors {
                            ui.label(RichText::new(err.to_string()).color(Color32::from_rgb(231, 76, 60)));
                        }
                    }
                });
                return;
            }

            // Top Bar: Connection, Theme, Account, Settings, Debug, Simulated Mode, Help
            ui.horizontal(|ui| {
                // Connection Status
                let (status_text, status_color, icon_text) = match self.connection_status {
                    ConnectionStatus::Connected => (
                        "Connected to IBKR",
                        Color32::from_rgb(46, 204, 113),
                        egui_phosphor::phosphor::CHECK_CIRCLE_BOLD,
                    ),
                    ConnectionStatus::Connecting => (
                        "Connecting...",
                        Color32::YELLOW,
                        egui_phosphor::phosphor::CIRCLE_DASHED_BOLD,
                    ),
                    ConnectionStatus::Disconnected => (
                        "Disconnected",
                        Color32::from_rgb(231, 76, 60),
                        egui_phosphor::phosphor::X_CIRCLE_BOLD,
                    ),
                };
                if matches!(self.connection_status, ConnectionStatus::Connecting) {
                    ui.spinner();
                    ui.add_space(4.0);
                }
                ui.label(
                    egui::RichText::new(egui_phosphor::phosphor_icon(icon_text))
                        .color(status_color)
                        .size(18.0),
                );
                ui.add_space(4.0);
                ui.label(
                    egui::RichText::new(status_text)
                        .color(status_color)
                        .strong()
                );

                // Settings (Admin only) 
                if self.user_session.role == UserRole::Admin {
                    if ui
                        .add(egui::Button::new(egui_phosphor::phosphor_icon(egui_phosphor::phosphor::GEAR_BOLD))
                            .text("Settings"))
                        .on_hover_text("Configure IBKR Host/Port/Account")
                        .clicked()
                    {
                        ui.memory_mut(|mem| mem.toggle_popup("ibkr_settings"));
                    }
                    egui::popup::popup_below_widget(ui, "ibkr_settings", ui.button(""), |ui| {
                        ui.heading("IBKR Settings");
                        ui.separator();
                        ui.label("Host:");
                        ui.text_edit_singleline(&mut self.ibkr_config.host);
                        ui.label("Port:");
                        ui.add(DragValue::new(&mut self.ibkr_config.port).clamp_range(1..=65535));
                        ui.label("Timeout (secs):");
                        ui.add(DragValue::new(&mut self.ibkr_config.timeout_secs).clamp_range(1..=30));
                        ui.label("Account ID:");
                        ui.text_edit_singleline(&mut self.ibkr_config.account_id);
                        ui.add_space(8.0);
                        if ui.button("Connect").clicked() {
                            self.connect_to_ibkr();
                            ui.close_menu();
                        }
                    });
                }

                // Theme Toggle 
                if ui
                    .add(egui::Button::new(
                        egui_phosphor::phosphor_icon(if self.dark_mode {
                            egui_phosphor::phosphor::SUN_BOLD
                        } else {
                            egui_phosphor::phosphor::MOON_BOLD
                        })
                    ).text(if self.dark_mode { "Light Mode" } else { "Dark Mode" }))
                    .on_hover_text("Toggle dark/light mode")
                    .clicked()
                {
                    self.dark_mode = !self.dark_mode;
                    self.save_settings();
                }

                // Account Switch 
                if self.user_session.role != UserRole::Observer && self.accounts.len() > 1 {
                    if ui
                        .add(egui::Button::new(egui_phosphor::phosphor_icon(egui_phosphor::phosphor::ARROWS_CLOCKWISE_BOLD))
                            .text("Switch Account"))
                        .on_hover_text("Switch between multiple IBKR accounts")
                        .clicked()
                    {
                        self.selected_account = (self.selected_account + 1) % self.accounts.len();
                        self.ibkr_config = self.accounts[self.selected_account].clone();
                        self.save_settings();
                    }
                }

                // Debug Toggle 
                if ui
                    .add(egui::Button::new(egui_phosphor::phosphor_icon(egui_phosphor::phosphor::BUG_BOLD))
                        .text("Debug"))
                    .on_hover_text("Toggle debug mode")
                    .clicked()
                {
                    self.debug_mode = !self.debug_mode;
                }

                // Simulated Mode Toggle 
                if ui
                    .add(egui::Button::new(
                        egui_phosphor::phosphor_icon(if self.simulated_mode {
                            egui_phosphor::phosphor::PLAY_BOLD
                        } else {
                            egui_phosphor::phosphor::PAUSE_BOLD
                        })
                    ).text(if self.simulated_mode { "Simulated: ON" } else { "Simulated: OFF" }))
                    .on_hover_text("Toggle simulated trading mode (virtual funds, no real orders)")
                    .clicked()
                {
                    self.simulated_mode = !self.simulated_mode;
                    self.save_settings();
                }
            });
                if ui.button("Help").on_hover_text("Show help and documentation").clicked() {
                    ui.memory_mut(|mem| mem.toggle_popup("help_popup"));
                }
                egui::popup::popup_below_widget(ui, "help_popup", ui.button(""), |ui| {
                    ui.heading("Help & Documentation");
                    ui.label("• Drag and drop order book entries to adjust price/size.");
                    ui.label("• Use the chart to view price history.");
                    ui.label("• Hover over fields for tooltips.");
                    ui.label("• Advanced order types: Trailing Stop, FOK, IOC, Iceberg, etc.");
                    ui.label("• Session will warn before timeout, click 'Extend Session' to stay logged in.");
                    ui.label("• For mobile/web, use the web version or resize the window for responsive layout.");
                });
            });

            // Charting: Price History 
            ui.group(|ui| {
                ui.heading("Market Price Chart");
                if self.price_history.is_empty() {
                    ui.label("No price data available.");
                } else {
                    // Convert price history to plot points
                    let points: egui::plot::PlotPoints = egui::plot::PlotPoints::from_iter(
                        self.price_history.iter().enumerate().map(|(i, (_, price))| {
                            [i as f64, price.to_f64().unwrap_or(0.0)]
                        })
                    );

                    // Calculate Y-axis bounds
                    let (min_y, max_y) = {
                        let ys = self.price_history.iter().map(|(_, p)| p.to_f64().unwrap_or(0.0));
                        let min = ys.clone().fold(f64::INFINITY, f64::min);
                        let max = ys.fold(f64::NEG_INFINITY, f64::max);
                        (min, max)
                    };

                    // Create styled line series
                    let line = egui::plot::Line::new(points.clone())
                        .name("Price")
                        .color(egui::Color32::LIGHT_BLUE)
                        .width(2.0)
                        .style(egui::plot::LineStyle::Solid);

                    // Plot with axes, grid, zoom & pan enabled
                    egui::plot::Plot::new("price_history_plot")
                        .height(200.0)
                        .legend(egui::plot::Legend::default().position(egui::plot::Corner::TopLeft))
                        .view_aspect(2.0)
                        .include_x(0.0)
                        .include_x(self.price_history.len() as f64)
                        .include_y(min_y)
                        .include_y(max_y)
                        .allow_drag(true)
                        .allow_zoom(true)
                        .show(ui, |plot_ui| {
                            // draw grid lines
                            plot_ui.grid(
                                egui::plot::Grid::default()
                                    .num_x_lines(5)
                                    .num_y_lines(5)
                            );
                            // draw the line
                            plot_ui.line(line.clone());
                            // draw points on the line
                            plot_ui.points(
                                egui::plot::Points::new(points)
                                    .radius(3.0)
                                    .fill(egui::Color32::WHITE),
                            );
                        });
                }
            });
            // Symbol, Quantity, Price, Order Type, Duration Entry with Real-Time Error Display and Custom Orders 
            ui.horizontal(|ui| {
                ui.label("Symbol:").on_hover_text("Stock symbol, e.g. AAPL");
                let symbol_edit = ui.add(TextEdit::singleline(&mut self.symbol)
                    .hint_text("Enter stock symbol (e.g. AAPL)"));
                if let Some(errors) = self.error_fields.get("symbol") {
                    for err in errors {
                        symbol_edit.highlight_with_color(Color32::from_rgb(231, 76, 60));
                        ui.label(RichText::new(err.to_string()).color(Color32::from_rgb(231, 76, 60)))
                            .on_hover_text("Symbol is required and must be valid.");
                    }
                }
            });

            ui.horizontal(|ui| {
                ui.add_space(10.0);
                ui.label("Quantity:").on_hover_text("Enter a positive number of units (min 1, max 10,000)");
                let quantity_edit = ui.add(TextEdit::singleline(&mut self.quantity)
                    .hint_text("Enter quantity")
                    .desired_width(100.0));
                if let Some(errors) = self.error_fields.get("quantity") {
                    for err in errors {
                        quantity_edit.highlight_with_color(Color32::from_rgb(231, 76, 60));
                        ui.label(RichText::new(err.to_string()).color(Color32::from_rgb(231, 76, 60)))
                            .on_hover_text("Check available funds and order size limits.");
                    }
                }
                ui.add_space(20.0);
                ui.label("Price:").on_hover_text(format!(
                    "Enter a positive price per unit (tick size ${:.2}, min ${:.2}, max ${:.2})",
                    self.min_tick_size, self.min_price, self.max_price
                ));
                let price_edit = ui.add(TextEdit::singleline(&mut self.price)
                    .hint_text("Enter price")
                    .desired_width(100.0));
                if let Some(errors) = self.error_fields.get("price") {
                    for err in errors {
                        price_edit.highlight_with_color(Color32::from_rgb(231, 76, 60));
                        ui.label(RichText::new(err.to_string()).color(Color32::from_rgb(231, 76, 60)))
                            .on_hover_text("Price must be within allowed range and tick size.");
                    }
                }
                ui.add_space(20.0);
                ui.label("Order Type:").on_hover_text("Choose order type: Limit, Market, Stop, Stop Limit, Trailing Stop, Iceberg Limit, Market with Limit, Simulated, FOK, IOC");
                let allowed_types = match self.user_session.role {
                    UserRole::Admin => vec![
                        "Limit", "Market", "Stop", "Stop Limit", "Trailing Stop",
                        "Iceberg Limit", "Market with Limit", "Simulated", "FOK", "IOC"
                    ],
                    UserRole::Trader => vec![
                        "Limit", "Market", "Stop", "Stop Limit",
                        "Iceberg Limit", "Market with Limit", "Simulated", "FOK", "IOC"
                    ],
                    UserRole::Observer => vec![],
                };
                ComboBox::from_id_source("order_type")
                    .selected_text(&self.order_type)
                    .show_ui(ui, |ui| {
                        for t in &allowed_types {
                            ui.selectable_value(&mut self.order_type, t.to_string(), *t);
                        }
                    });
                if let Some(errors) = self.error_fields.get("order_type") {
                    for err in errors {
                        ui.label(RichText::new(err.to_string()).color(Color32::from_rgb(231, 76, 60)))
                            .on_hover_text("Order type not allowed for your role.");
                    }
                }
                ui.add_space(20.0);
                ui.label("Duration:").on_hover_text("Order duration: Day, GTC, FOK, IOC");
                ComboBox::from_id_source("order_duration")
                    .selected_text(&self.order_duration)
                    .show_ui(ui, |ui| {
                        for t in ["Day", "GTC", "FOK", "IOC"] {
                            ui.selectable_value(&mut self.order_duration, t.to_string(), t);
                        }
                    });
            });

            // Custom Order Parameters UI 
            if self.order_type == "Iceberg Limit" {
                ui.horizontal(|ui| {
                    ui.label("Iceberg Size:").on_hover_text("Visible portion of the order");
                    let mut iceberg_str = self.custom_order_params.iceberg_size.map(|v| v.to_string()).unwrap_or_default();
                    if ui.text_edit_singleline(&mut iceberg_str).changed() {
                        self.custom_order_params.iceberg_size = iceberg_str.parse::<Decimal>().ok();
                    }
                    if let Some(errors) = self.error_fields.get("iceberg_size") {
                        for err in errors {
                            ui.label(RichText::new(err.to_string()).color(Color32::from_rgb(231, 76, 60)));
                        }
                    }
                });
            }
            if self.order_type == "Market with Limit" {
                ui.horizontal(|ui| {
                    ui.label("Limit Offset:").on_hover_text("Offset from market price for limit");
                    let mut offset_str = self.custom_order_params.limit_offset.map(|v| v.to_string()).unwrap_or_default();
                    if ui.text_edit_singleline(&mut offset_str).changed() {
                        self.custom_order_params.limit_offset = offset_str.parse::<Decimal>().ok();
                    }
                    if let Some(errors) = self.error_fields.get("limit_offset") {
                        for err in errors {
                            ui.label(RichText::new(err.to_string()).color(Color32::from_rgb(231, 76, 60)));
                        }
                    }
                });
            }
            if self.order_type == "Trailing Stop" {
                ui.horizontal(|ui| {
                    ui.label("Trailing Amount:").on_hover_text("Trailing stop amount");
                    let mut trailing_str = self.custom_order_params.trailing_amount.map(|v| v.to_string()).unwrap_or_default();
                    if ui.text_edit_singleline(&mut trailing_str).changed() {
                        self.custom_order_params.trailing_amount = trailing_str.parse::<Decimal>().ok();
                    }
                });
            }
            if self.order_type == "FOK" {
                ui.horizontal(|ui| {
                    ui.checkbox(&mut self.custom_order_params.fok, "Fill-or-Kill (FOK)").on_hover_text("Order must be filled in its entirety or cancelled immediately.");
                });
            }
            if self.order_type == "IOC" {
                ui.horizontal(|ui| {
                    ui.checkbox(&mut self.custom_order_params.ioc, "Immediate-or-Cancel (IOC)").on_hover_text("Order will fill as much as possible immediately, rest is cancelled.");
                });
            }

            // Order Buttons with Confirmation Dialog, Retry, and Drag-and-Drop Support 
            if self.user_session.role != UserRole::Observer {
                ui.horizontal(|ui| {
                    fn create_button_style(color: Color32) -> egui::ButtonStyle {
                        egui::ButtonStyle {
                            background: Some(color),
                            text_color: Color32::WHITE,
                            shadow: Shadow {
                                extrusion: 5.0,
                                color: Color32::from_black_alpha(100),
                            },
                            ..Default::default()
                        }
                    }
                    ui.add_space(10.0);
                    let buy_style = create_button_style(Color32::from_rgb(46, 204, 113));
                    if ui.add(Button::new("Buy").style(buy_style)).clicked() {
                        ui.memory_mut(|mem| mem.toggle_popup("confirm_buy"));
                    }
                    egui::popup::popup_below_widget(ui, "confirm_buy", ui.button(""), |ui| {
                        ui.label("Confirm Buy Order?");
                        if ui.button("Confirm").clicked() {
                            self.place_order(true);
                            ui.close_menu();
                        }
                        if ui.button("Cancel").clicked() {
                            ui.close_menu();
                        }
                    });
                    ui.add_space(20.0);
                    let sell_style = create_button_style(Color32::from_rgb(231, 76, 60));
                    if ui.add(Button::new("Sell").style(sell_style)).clicked() {
                        ui.memory_mut(|mem| mem.toggle_popup("confirm_sell"));
                    }
                    egui::popup::popup_below_widget(ui, "confirm_sell", ui.button(""), |ui| {
                        ui.label("Confirm Sell Order?");
                        if ui.button("Confirm").clicked() {
                            self.place_order(false);
                            ui.close_menu();
                        }
                        if ui.button("Cancel").clicked() {
                            ui.close_menu();
                        }
                    });
                    ui.add_space(20.0);
                    let clear_style = create_button_style(Color32::from_rgb(100, 100, 100));
                    if ui.add(Button::new("Clear Order Book").style(clear_style)).clicked() {
                        self.order_book.clear();
                        self.status_message = "Order book cleared".to_string();
                        info!("Order book cleared by user.");
                        self.audit_trail.log(&format!(
                            "[{}] User '{}' (ip: {:?}) cleared order book",
                            Utc::now(), self.user_session.username, self.user_session.ip_address
                        ));
                    }
                    if !self.error_fields.is_empty() && ui.button("Retry Last Action").clicked() {
                        self.retry_last_action();
                    }
                });
            }

            // Error Summary Section 
            if !self.error_fields.is_empty() {
                ui.group(|ui| {
                    ui.label(RichText::new("Please fix the following errors:").color(Color32::from_rgb(231, 76, 60)));
                    for msg in self.error_fields.summary() {
                        ui.label(RichText::new(msg).color(Color32::from_rgb(231, 76, 60)));
                    }
                });
            }

            // Status Message and Message History 
            if !self.status_message.is_empty() && self.show_status {
                ui.group(|ui| {
                    ui.horizontal(|ui| {
                        let status_color = if self.status_message.contains("failed") || self.status_message.contains("Invalid") {
                            Color32::from_rgb(231, 76, 60)
                        } else if self.status_message.contains("success") || self.status_message.contains("filled") {
                            Color32::from_rgb(46, 204, 113)
                        } else {
                            Color32::WHITE
                        };
                        ui.label(RichText::new(&self.status_message).color(status_color));
                        if ui.small_button("Close").clicked() {
                            self.show_status = false;
                            self.error_fields.clear();
                        }
                    });
                    ui.collapsing("Message History", |ui| {
                        for (timestamp, message) in self.message_history.iter().rev().take(10) {
                            ui.label(format!("[{}] {}", timestamp.format("%H:%M:%S"), message));
                        }
                    });
                });
            }

            // DOM Display with Level Limiting, Scrolling, Tooltips, Color Cues, and Drag-and-Drop 
            ui.group(|ui| {
                ui.horizontal(|ui| {
                    ui.heading("Depth of Market");
                    ui.add_space(20.0);
                    if ui.button(if self.sort_by_price { "Sort by Size" } else { "Sort by Price" }).clicked() {
                        self.sort_by_price = !self.sort_by_price;
                    }
                    ui.add_space(20.0);
                    ui.label("DOM Levels:");
                    ui.add(DragValue::new(&mut self.dom_levels).clamp_range(1..=100));
                    ui.add_space(20.0);
                    if ui.button("Load More Levels").clicked() {
                        self.dom_levels = (self.dom_levels + 10).min(100);
                    }
                });

                ScrollArea::vertical().show(ui, |ui| {
                    ui.columns(2, |columns| {
                        // Ask Side 
                        columns[0].group(|ui| {
                            ui.heading("Ask Orders");
                            ui.horizontal(|ui| {
                                ui.label("Price").min_width(120.0).on_hover_text("Ask price (lowest = best ask)");
                                ui.label("Size").min_width(100.0).on_hover_text("Total size at this price");
                                ui.label("Value").min_width(100.0).on_hover_text("Price × Size");
                                ui.label("%").min_width(80.0).on_hover_text("Percent of total ask size");
                            });
                            ui.separator();

                            let asks: Vec<_> = self.order_book.asks.iter().collect();
                            let total_ask_size: Decimal = asks.iter().map(|(_, size)| *size).sum();
                            let total_ask_value: Decimal = asks.iter().map(|(price, size)| price.0 * *size).sum();

                            let mut asks_sorted = asks.clone();
                            if self.sort_by_price {
                                asks_sorted.sort_by(|a, b| b.0.0.cmp(&a.0.0));
                            } else {
                                asks_sorted.sort_by(|a, b| b.1.cmp(&a.1));
                            }
                            let best_ask = self.order_book.get_best_ask();
                            let avg_ask = self.order_book.get_average_ask_price();

                            for (i, (price, size)) in asks_sorted.iter().take(self.dom_levels).enumerate() {
                                let value = price.0 * **size;
                                let percentage = if total_ask_size > Decimal::ZERO {
                                    (**size / total_ask_size) * Decimal::from(100)
                                } else {
                                    Decimal::ZERO
                                };
                                let row_color = if Some(price.0) == best_ask {
                                    Color32::from_rgb(231, 76, 60)
                                } else if i == 0 {
                                    Color32::from_rgb(255, 200, 200)
                                } else {
                                    Color32::WHITE
                                };
                                // Drag-and-drop support for ask orders
                                let response = ui.horizontal(|ui| {
                                    ui.label(RichText::new(format!("${:.2}", price.0)).color(row_color))
                                        .min_width(120.0)
                                        .on_hover_text(format!("Number of orders at this price: {}", self.order_book.total_ask_orders));
                                    ui.label(RichText::new(format!("{}", size)).color(row_color)).min_width(100.0);
                                    ui.label(RichText::new(format!("${:.2}", value)).color(row_color)).min_width(100.0);
                                    ui.label(RichText::new(format!("{:.1}%", percentage)).color(row_color)).min_width(80.0);
                                });
                                let id = egui::Id::new(format!("ask_drag_{}_{}", price.0, size));
                                let drag = response.response.interact(egui::Sense::drag());
                                if drag.drag_started() {
                                    self.drag_order = Some((false, price.0, **size));
                                }
                                if drag.drag_released() {
                                    if let Some((is_bid, p, s)) = self.drag_order.take() {
                                        // On drop, allow user to edit or move order
                                        ui.memory_mut(|mem| mem.toggle_popup("edit_order_popup"));
                                        // Store drag info in app state for popup
                                    }
                                }
                            }
                            if asks_sorted.len() > self.dom_levels {
                                ui.label(format!("... ({} more ask levels)", asks_sorted.len() - self.dom_levels));
                            }
                            ui.separator();
                            ui.label(format!("Total Ask Size: {}", total_ask_size));
                            ui.label(format!("Total Ask Value: ${:.2}", total_ask_value));
                            if let Some(avg) = avg_ask {
                                ui.label(format!("Average Ask Price: ${:.2}", avg));
                            }
                            ui.label(format!("Total Ask Orders: {}", self.order_book.total_ask_orders));
                        });

                        // Bid Side
                        columns[1].group(|ui| {
                            ui.heading("Bid Orders");
                            ui.horizontal(|ui| {
                                ui.label("Price").min_width(120.0).on_hover_text("Bid price (highest = best bid)");
                                ui.label("Size").min_width(100.0).on_hover_text("Total size at this price");
                                ui.label("Value").min_width(100.0).on_hover_text("Price × Size");
                                ui.label("%").min_width(80.0).on_hover_text("Percent of total bid size");
                            });
                            ui.separator();

                            let bids: Vec<_> = self.order_book.bids.iter().collect();
                            let total_bid_size: Decimal = bids.iter().map(|(_, size)| *size).sum();
                            let total_bid_value: Decimal = bids.iter().map(|(price, size)| *price * *size).sum();

                            let mut bids_sorted = bids.clone();
                            if self.sort_by_price {
                                bids_sorted.sort_by(|a, b| b.0.cmp(&a.0));
                            } else {
                                bids_sorted.sort_by(|a, b| b.1.cmp(&a.1));
                            }
                            let best_bid = self.order_book.get_best_bid();
                            let avg_bid = self.order_book.get_average_bid_price();

                            for (i, (price, size)) in bids_sorted.iter().take(self.dom_levels).enumerate() {
                                let value = **price * **size;
                                let percentage = if total_bid_size > Decimal::ZERO {
                                    (**size / total_bid_size) * Decimal::from(100)
                                } else {
                                    Decimal::ZERO
                                };
                                let row_color = if Some(**price) == best_bid {
                                    Color32::from_rgb(46, 204, 113)
                                } else if i == 0 {
                                    Color32::from_rgb(200, 255, 200)
                                } else {
                                    Color32::WHITE
                                };
                                // Drag-and-drop support for bid orders
                                let response = ui.horizontal(|ui| {
                                    ui.label(RichText::new(format!("${:.2}", price)).color(row_color))
                                        .min_width(120.0)
                                        .on_hover_text(format!("Number of orders at this price: {}", self.order_book.total_bid_orders));
                                    ui.label(RichText::new(format!("{}", size)).color(row_color)).min_width(100.0);
                                    ui.label(RichText::new(format!("${:.2}", value)).color(row_color)).min_width(100.0);
                                    ui.label(RichText::new(format!("{:.1}%", percentage)).color(row_color)).min_width(80.0);
                                });
                                let id = egui::Id::new(format!("bid_drag_{}_{}", price, size));
                                let drag = response.response.interact(egui::Sense::drag());
                                if drag.drag_started() {
                                    self.drag_order = Some((true, **price, **size));
                                }
                                if drag.drag_released() {
                                    if let Some((is_bid, p, s)) = self.drag_order.take() {
                                        ui.memory_mut(|mem| mem.toggle_popup("edit_order_popup"));
                                    }
                                }
                            }
                            if bids_sorted.len() > self.dom_levels {
                                ui.label(format!("... ({} more bid levels)", bids_sorted.len() - self.dom_levels));
                            }
                            ui.separator();
                            ui.label(format!("Total Bid Size: {}", total_bid_size));
                            ui.label(format!("Total Bid Value: ${:.2}", total_bid_value));
                            if let Some(avg) = avg_bid {
                                ui.label(format!("Average Bid Price: ${:.2}", avg));
                            }
                            ui.label(format!("Total Bid Orders: {}", self.order_book.total_bid_orders));
                        });
                    });
                });
            });

            // Order History (Audit Trail) 
            ui.collapsing("Order History", |ui| {
                for entry in self.order_history.iter().rev().take(20) {
                    ui.label(format!(
                        "[{}] {}: {} {} {} @ ${} ({}, {}) [{}]",
                        entry.timestamp.format("%Y-%m-%d %H:%M:%S"),
                        entry.user,
                        entry.action,
                        entry.size,
                        entry.symbol,
                        entry.price,
                        entry.order_type,
                        entry.duration,
                        entry.result
                    ));
                }
            });

            // Real-Time Order Status Feedback and Risk Alerts 
            ui.collapsing("Order Statuses (Real-Time)", |ui| {
                for (order_id, status) in self.order_statuses.iter() {
                    let status_str = match status {
                        OrderStatus::Pending => "Pending".to_string(),
                        OrderStatus::Filled => "Filled".to_string(),
                        OrderStatus::PartiallyFilled(qty) => format!("Partially Filled: {}", qty),
                        OrderStatus::Cancelled => "Cancelled".to_string(),
                        OrderStatus::Rejected(reason) => format!("Rejected: {}", reason),
                        OrderStatus::FOKFailed => "FOK: Not filled, cancelled".to_string(),
                        OrderStatus::IOCPartial(qty) => format!("IOC: Partial fill: {}", qty),
                    };
                    ui.label(format!("Order {}: {}", order_id, status_str));
                }
                if !self.risk_alerts.is_empty() {
                    ui.label(RichText::new("Risk Alerts:").color(Color32::YELLOW));
                    for alert in &self.risk_alerts {
                        ui.label(RichText::new(alert).color(Color32::YELLOW));
                    }
                }
            });

            // Backtesting Results 
            if let Some(backtest) = &self.backtest_results {
                ui.group(|ui| {
                    ui.heading("Backtest Results");
                    ui.label(format!("PnL: ${:.2}", backtest.pnl));
                    ui.label(format!("Max Drawdown: ${:.2}", backtest.max_drawdown));
                    ui.label(format!("Sharpe Ratio: {:.2}", backtest.sharpe));
                    ui.collapsing("Backtest Trades", |ui| {
                        for entry in &backtest.trades {
                            ui.label(format!(
                                "[{}] {} {} @ ${} ({})",
                                entry.timestamp.format("%Y-%m-%d %H:%M:%S"),
                                entry.action,
                                entry.size,
                                entry.price,
                                entry.result
                            ));
                        }
                    });
                });
            }

            // Debug Mode: Show Internal State 
            if self.debug_mode {
                ui.group(|ui| {
                    ui.label(RichText::new("Debug Info").color(Color32::YELLOW));
                    ui.label(format!("Symbol: {}", self.symbol));
                    ui.label(format!("Quantity: {}", self.quantity));
                    ui.label(format!("Price: {}", self.price));
                    ui.label(format!("Order Type: {}", self.order_type));
                    ui.label(format!("Order Duration: {}", self.order_duration));
                    ui.label(format!("Connection Status: {:?}", self.connection_status));
                    ui.label(format!("Order Book: {:?}", self.order_book));
                    ui.label(format!("User Session: {:?}", self.user_session));
                    ui.label(format!("IBKR Config: {:?}", self.ibkr_config));
                    ui.label(format!("Persistent Settings: {:?}", self.persistent_settings));
                    ui.label(format!("Simulated Mode: {}", self.simulated_mode));
                });
            }
        });
    }
}

// Entry point for the DOM Trading System application.
fn main() -> eframe::Result<()> {
    // Initialize the logger only if not already set, to avoid double initialization in tests or integration.
    let _ = env_logger::try_init();

    let native_options = eframe::NativeOptions {
        drag_and_drop_support: true,
        follow_system_theme: true,
        maximized: false,
        initial_window_size: Some(Vec2::new(480.0, 800.0)),
        ..Default::default()
    };

    eframe::run_native(
        "DOM Trading System",
        native_options,
        Box::new(|cc| Box::new(TradingApp::new(cc))),
    )
}

// Returns all non-loopback IP addresses using `local_ip_address`.
fn list_non_loopback_ips() -> Vec<IpAddr> {
    match local_ip_address::list_afinet_netifas() {
        Ok(interfaces) => interfaces
            .into_iter()
            .map(|(_ifname, ip)| ip)
            .filter(|ip| !ip.is_loopback())
            .collect(),
        Err(_) => Vec::new(),
    }
}
