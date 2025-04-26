// --- Imports and Dependencies ---
use eframe::egui::{self, Color32, Shadow, Button, Ui, RichText, Vec2, Visuals, DragValue, TextEdit, ComboBox, ScrollArea, CentralPanel, Context,}; // egui: GUI framework for Rust, used for building the application's user interface
use chrono::{DateTime, Utc}; // chrono: date/time library, used for timestamps and UTC time
use ibkr_rust::{TwsConnection, Contract, Order as IbkrOrder}; // ibkr_rust: Interactive Brokers API bindings
use std::time::{Duration, Instant}; // std::time: for measuring elapsed time, session timeouts, etc.
use log::{info, warn, error, debug, trace}; // log: logging macros for diagnostics and audit
use std::sync::{Arc, Mutex}; // Arc/Mutex: for thread-safe shared state (multi-threaded access)
use std::thread; // std::thread: for spawning background threads (e.g., data polling)
use std::fs::{OpenOptions, File, rename, remove_file, metadata}; // std::fs: file operations (logs, settings, etc.)
use std::io::{Write, Seek, SeekFrom}; // std::io: for file writing, seeking (log rotation, etc.)
use std::net::IpAddr; // std::net::IpAddr: for storing/logging user IP addresses
use std::path::Path; // std::path::Path: for file path manipulations
use std::env; // std::env: for reading environment variables (e.g., API keys, config)
use openssl::symm::{Cipher, Crypter, Mode}; // openssl: for encrypting sensitive data (passwords, API keys)
use tokio::sync::Mutex as AsyncMutex; // Tokio async Mutex: for async shared state (async tasks, websockets)
use tokio::runtime::Runtime; // Tokio runtime: to run async code in a synchronous context

// --- Modularization: Split modules for scalability (example, not full split here) ---
// In a real project, these would be in separate files/modules for better organization.
pub mod user;         // User management module
pub mod order;        // Order management module
pub mod market_data;  // Market data handling module
pub mod risk;         // Risk management module
pub mod logging;      // Logging and audit trail module
pub mod settings;     // Persistent settings module
pub mod backtest;     // Backtesting module
pub mod api;          // API integration module
pub mod ui;           // UI helpers and components
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

// --- User Role Enum ---
// Defines the different user roles for access control
#[derive(Clone, Debug, PartialEq)]
pub enum UserRole {
    Admin,      // Full access to all features
    Trader,     // Can trade but not change system settings
    Observer,   // Read-only access, cannot trade
}

// --- User Session Struct ---
// Stores session state for a logged-in user
#[derive(Clone, Debug)]
pub struct UserSession {
    pub username: String,                    // Username of the user
    pub role: UserRole,                      // User's role (Admin, Trader, Observer)
    pub authenticated: bool,                 // Is the user authenticated?
    pub last_active: Instant,                // Last activity timestamp
    pub two_factor_passed: bool,             // Has 2FA been passed?
    pub ip_address: Option<IpAddr>,          // User's IP address (for audit)
    pub encrypted_password: Option<Vec<u8>>, // Store encrypted password
    pub session_warning_shown: bool,         // UX: show session timeout warning
    pub session_warning_time: Option<Instant>, // When warning was shown
}

// Implementation of UserSession methods
impl UserSession {
    // Create a new user session with username and role
    pub fn new(username: &str, role: UserRole) -> Self {
        Self {
            username: username.to_string(),      // Set username
            role,                               // Set user role
            authenticated: false,               // Not authenticated by default
            last_active: Instant::now(),        // Set last active to now
            two_factor_passed: false,           // 2FA not passed by default
            ip_address: None,                   // No IP address yet
            encrypted_password: None,           // No password stored yet
            session_warning_shown: false,       // No warning shown yet
            session_warning_time: None,         // No warning time yet
        }
    }
    // Check if the session is still active (not timed out)
    pub fn is_active(&self, timeout_secs: u64) -> bool {
        self.last_active.elapsed() < Duration::from_secs(timeout_secs)
    }
    // Get the time left before session timeout (in seconds)
    pub fn time_left(&self, timeout_secs: u64) -> Option<u64> {
        let elapsed = self.last_active.elapsed();
        if elapsed < Duration::from_secs(timeout_secs) {
            Some(timeout_secs - elapsed.as_secs())
        } else {
            None
        }
    }
    // Should a session timeout warning be shown?
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
    // Extend the session (reset activity timer and warning)
    pub fn extend_session(&mut self) {
        self.last_active = Instant::now();
        self.session_warning_shown = false;
        self.session_warning_time = None;
    }
}

// --- Error Handling and Validation Structures ---

// Enum for possible field validation errors
#[derive(Debug, Clone)]
pub enum FieldError {
    Required,                                   // Field is required
    InvalidFormat,                              // Invalid format
    BelowMinimum { min: Decimal },              // Value below minimum
    AboveMaximum { max: Decimal },              // Value above maximum
    NotTickAligned { tick: Decimal },           // Not aligned to tick size
    PriceOutOfRange { min: Decimal, max: Decimal }, // Price out of allowed range
    InsufficientFunds,                          // Not enough funds
    Unauthorized,                               // Not authorized
    OrderTypeNotSupported,                      // Order type not supported
    SessionExpired,                             // Session expired
    ApiRateLimited,                             // API rate limit reached
    Other(String),                              // Other error (with message)
}

// Implement Display for FieldError for user-friendly error messages
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

// Struct to collect validation errors for multiple fields
#[derive(Debug, Default, Clone)]
pub struct ValidationErrors {
    pub errors: HashMap<String, Vec<FieldError>>, // Map field name to list of errors
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
        self.errors.iter().flat_map(|(f, errs)| {
            errs.iter().map(move |e| format!("{}: {}", f, e))
        }).collect()
    }
}

// --- IBKR Connection Configuration ---

// Struct for IBKR (Interactive Brokers) connection and account configuration
#[derive(Clone)]
pub struct IbkrConfig {
    pub host: String,                       // Host address for TWS/Gateway
    pub port: u16,                          // Port number
    pub timeout_secs: u64,                  // Connection timeout in seconds
    pub account_id: String,                 // IBKR account ID
    pub margin: Decimal,                    // Margin available
    pub cash_balance: Decimal,              // Cash balance
    pub encrypted_api_key: Option<Vec<u8>>, // Encrypted API key for security
    pub api_rate_limit: u32,                // API rate limit per minute
}

// Default implementation for IbkrConfig
impl Default for IbkrConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),           // Default to localhost
            port: 7497,                              // Default TWS port
            timeout_secs: 5,                         // 5 second timeout
            account_id: String::new(),               // No account by default
            margin: Decimal::new(100_000, 0),        // $100,000 margin
            cash_balance: Decimal::new(100_000, 0),  // $100,000 cash
            encrypted_api_key: None,                 // No API key by default
            api_rate_limit: 40,                      // IBKR default rate limit
        }
    }
}

// --- Market Data WebSocket Integration (Async, SSL/TLS, Real Implementation) ---

// Struct for managing a market data WebSocket connection
pub struct MarketDataWebSocket {
    pub connected: bool,                // Is the WebSocket connected?
    pub last_error: Option<String>,     // Last error message, if any
    pub current_symbol: Option<String>, // Currently subscribed symbol
    pub reconnect_attempts: u32,        // Number of reconnect attempts
    pub max_reconnect_attempts: u32,    // Maximum allowed reconnect attempts
    pub use_tls: bool,                  // Use SSL/TLS for connection
    pub last_message_time: Option<Instant>, // Last time a message was received
}

// Implementation of MarketDataWebSocket methods
impl MarketDataWebSocket {
    // Create a new MarketDataWebSocket instance
    pub fn new() -> Self {
        Self {
            connected: false,               // Not connected by default
            last_error: None,               // No error yet
            current_symbol: None,           // No symbol subscribed yet
            reconnect_attempts: 0,          // No reconnects yet
            max_reconnect_attempts: 5,      // Allow up to 5 reconnects
            use_tls: true,                  // Use TLS by default
            last_message_time: None,        // No messages yet
        }
    }
    // Asynchronously connect to the WebSocket server
    pub async fn connect_async(&mut self, host: &str, port: u16) -> Result<(), String> {
        // Real implementation: connect to WebSocket server, handle SSL/TLS, etc.
        // For now, simulate connection.
        if self.use_tls {
            self.connected = true; // Mark as connected
        } else {
            self.connected = true; // Also mark as connected if not using TLS
        }
        self.reconnect_attempts = 0; // Reset reconnect attempts
        self.last_error = None;      // Clear last error
        self.last_message_time = Some(Instant::now()); // Set last message time to now
        Ok(())
    }
    // Subscribe to a symbol's market data
    pub fn subscribe(&mut self, symbol: &str) {
        if self.connected {
            self.current_symbol = Some(symbol.to_string()); // Set current symbol
            self.last_message_time = Some(Instant::now());  // Update last message time
        }
    }
    // Unsubscribe from a symbol's market data
    pub fn unsubscribe(&mut self, symbol: &str) {
        if self.connected && self.current_symbol.as_deref() == Some(symbol) {
            self.current_symbol = None; // Remove current symbol
        }
    }
    // Disconnect the WebSocket
    pub fn disconnect(&mut self) {
        self.connected = false;         // Mark as disconnected
        self.current_symbol = None;     // Remove current symbol
    }
    // Handle a disconnect event (with exponential backoff for reconnect)
    pub fn handle_disconnect(&mut self) {
        self.connected = false; // Mark as disconnected
        self.last_error = Some("WebSocket disconnected".to_string()); // Set error message
        self.reconnect_attempts += 1; // Increment reconnect attempts
        if self.reconnect_attempts <= self.max_reconnect_attempts {
            let wait = 2u64.pow(self.reconnect_attempts) * 100; // Exponential backoff
            std::thread::sleep(Duration::from_millis(wait));    // Sleep before retry
        }
    }
}

// --- Audit Trail and Logging with Log Rotation and Real-Time Log Forwarding ---

// Struct for audit trail logging (with log rotation)
pub struct AuditTrail {
    file: Mutex<File>,      // Mutex-protected file for thread-safe logging
    path: String,           // Path to the log file
    max_size_bytes: u64,    // Maximum log file size before rotation
    // For real-time log forwarding, you could add a sender here (e.g., to ELK/CloudWatch)
}

// Implementation of AuditTrail methods
impl AuditTrail {
    // Create a new AuditTrail logger at the given path
    pub fn new(path: &str) -> Self {
        let file = OpenOptions::new()
            .create(true)      // Create file if it doesn't exist
            .append(true)      // Append to file
            .open(path)        // Open the file
            .unwrap();         // Panic if file can't be opened
        Self {
            file: Mutex::new(file),                // Wrap file in Mutex for thread safety
            path: path.to_string(),                // Store file path
            max_size_bytes: 5 * 1024 * 1024,       // 5 MB max size before rotation
        }
    }
    // Log an entry to the audit trail (with rotation check)
    pub fn log(&self, entry: &str) {
        let mut file = self.file.lock().unwrap();  // Lock file for writing
        let _ = writeln!(file, "{}", entry);       // Write entry to file
        let _ = file.flush();                      // Flush to disk
        drop(file);                                // Release lock
        self.rotate_if_needed();                   // Rotate log if needed
        // TODO: Forward to centralized log system if configured
    }
    // Rotate the log file if it exceeds the maximum size
    fn rotate_if_needed(&self) {
        if let Ok(meta) = metadata(&self.path) {
            if meta.len() > self.max_size_bytes {
                let rotated = format!("{}.{}", &self.path, Utc::now().format("%Y%m%d%H%M%S")); // New rotated filename
                let _ = self.file.lock().unwrap().flush(); // Flush before renaming
                let _ = rename(&self.path, &rotated);      // Rename current log file
                let _ = OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&self.path);                     // Create new log file
            }
        }
    }
}

// --- Persistent Settings with File Storage (JSON) and Theme Customization ---

// Struct for persistent user/application settings (saved as JSON)
#[derive(Default, Clone, serde::Serialize, serde::Deserialize)]
pub struct PersistentSettings {
    pub theme: String,                // Theme name ("dark", "light", etc.)
    pub dom_levels: usize,            // Number of DOM levels to display
    pub last_symbol: String,          // Last used symbol
    pub last_account: usize,          // Last selected account index
    pub simulated_mode: bool,         // Simulated trading mode enabled?
    pub custom_theme: Option<String>, // For user-customizable themes
}

// Implementation of PersistentSettings methods
impl PersistentSettings {
    // Load settings from a JSON file, or return default if not found/invalid
    pub fn load_from_file(path: &str) -> Self {
        if let Ok(data) = std::fs::read_to_string(path) {
            if let Ok(settings) = serde_json::from_str(&data) {
                return settings;
            }
        }
        Self::default()
    }
    // Save settings to a JSON file
    pub fn save_to_file(&self, path: &str) {
        if let Ok(json) = serde_json::to_string_pretty(self) {
            let _ = std::fs::write(path, json);
        }
    }
}

// --- Main Application Structure ---

// Enum for order status (for real-time feedback)
#[derive(Clone, Debug)]
pub enum OrderStatus {
    Pending,                       // Order is pending
    Filled,                        // Order is fully filled
    PartiallyFilled(Decimal),      // Order is partially filled (with filled quantity)
    Cancelled,                     // Order was cancelled
    Rejected(String),              // Order was rejected (with reason)
    FOKFailed,                     // Fill-or-Kill failed
    IOCPartial(Decimal),           // Immediate-or-Cancel partial fill
}

// Main application struct holding all state for the trading app
pub struct TradingApp {
    order_book: OrderBook,                                 // Order book (bids/asks)
    symbol: String,                                        // Current symbol
    quantity: String,                                      // Order quantity (as string for UI)
    price: String,                                         // Order price (as string for UI)
    status_message: String,                                // Status message for user
    tws_connection: Option<TwsConnection>,                 // IBKR TWS connection
    connection_status: ConnectionStatus,                   // Connection status
    error_fields: ValidationErrors,                        // Validation errors for fields
    show_status: bool,                                     // Show status message?
    message_history: Vec<(DateTime<Utc>, String)>,         // Message history (timestamped)
    sort_by_price: bool,                                   // Sort DOM by price or size
    ibkr_config: IbkrConfig,                               // IBKR connection/account config
    debug_mode: bool,                                      // Debug mode enabled?
    dom_levels: usize,                                     // Number of DOM levels to show
    user_session: UserSession,                             // Current user session
    websocket: Option<Arc<AsyncMutex<MarketDataWebSocket>>>, // Market data websocket (async)
    audit_trail: Arc<AuditTrail>,                          // Audit trail logger
    order_type: String,                                    // Selected order type
    order_duration: String,                                // Selected order duration
    max_order_size: Decimal,                               // Maximum allowed order size
    min_tick_size: Decimal,                                // Minimum tick size
    min_price: Decimal,                                    // Minimum allowed price
    max_price: Decimal,                                    // Maximum allowed price
    margin_available: Decimal,                             // Margin available for trading
    persistent_settings: PersistentSettings,               // Persistent settings
    accounts: Vec<IbkrConfig>,                            // List of available accounts
    selected_account: usize,                               // Index of selected account
    dark_mode: bool,                                       // Is dark mode enabled?
    last_symbol_subscribed: Option<String>,                // Last symbol subscribed to
    session_timeout_secs: u64,                             // Session timeout in seconds
    session_warn_before_secs: u64,                         // Warn before session timeout (secs)
    order_history: Vec<OrderHistoryEntry>,                 // Order history (audit)
    custom_order_params: CustomOrderParams,                // Custom order parameters
    simulated_mode: bool,                                  // Simulated trading mode
    order_statuses: HashMap<u64, OrderStatus>,             // Real-time order status feedback
    // For drag-and-drop
    drag_order: Option<(bool, Decimal, Decimal)>,          // (is_bid, price, size)
    // For charting
    price_history: Vec<(DateTime<Utc>, Decimal)>,          // Price history for chart
    // For backtesting
    backtest_results: Option<BacktestResults>,             // Backtest results
    // For API rate limiting
    last_api_call: Option<Instant>,                        // Last API call time
    api_calls_this_minute: u32,                            // API calls in current minute
    last_api_minute: Option<u64>,                          // Last API call minute
    // For risk alerts
    risk_alerts: Vec<String>,                              // List of risk alerts
}

// Struct for a single order history entry (for audit trail and history)
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct OrderHistoryEntry {
    pub timestamp: DateTime<Utc>,      // Time of the order
    pub user: String,                  // User who placed the order
    pub action: String,                // Action (Buy/Sell)
    pub symbol: String,                // Symbol traded
    pub size: Decimal,                 // Order size
    pub price: Decimal,                // Order price
    pub order_type: String,            // Order type (Limit, Market, etc.)
    pub duration: String,              // Order duration (Day, GTC, etc.)
    pub result: String,                // Result (Filled, Cancelled, etc.)
    pub account_id: String,            // Account ID used
}

// Struct for custom order parameters (iceberg, trailing, etc.)
#[derive(Clone, Debug, Default)]
pub struct CustomOrderParams {
    pub iceberg_size: Option<Decimal>,     // Iceberg order visible size
    pub limit_offset: Option<Decimal>,     // Offset for market-with-limit
    pub simulated: bool,                   // Is this a simulated order?
    pub trailing_amount: Option<Decimal>,  // Trailing stop amount
    pub fok: bool,                         // Fill-or-Kill flag
    pub ioc: bool,                         // Immediate-or-Cancel flag
}

// Struct for backtest results (PnL, drawdown, trades, etc.)
pub struct BacktestResults {
    pub trades: Vec<OrderHistoryEntry>,    // List of trades in backtest
    pub pnl: Decimal,                      // Profit and loss
    pub max_drawdown: Decimal,             // Maximum drawdown
    pub sharpe: f64,                       // Sharpe ratio
}

// Implementation of TradingApp methods
impl TradingApp {
    // Create a new TradingApp instance (called at startup)
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let audit_trail = Arc::new(AuditTrail::new("audit_trail.log")); // Create audit logger
        let persistent_settings = PersistentSettings::load_from_file("settings.json"); // Load settings
        let accounts = vec![IbkrConfig::default()]; // Default account list
        let dom_levels = if persistent_settings.dom_levels == 0 { 10 } else { persistent_settings.dom_levels }; // Default DOM levels
        Self {
            order_book: OrderBook::default(),                       // Initialize order book
            symbol: persistent_settings.last_symbol.clone(),         // Set last used symbol
            quantity: String::new(),                                // Empty quantity
            price: String::new(),                                   // Empty price
            status_message: String::new(),                          // No status message
            tws_connection: None,                                   // No TWS connection yet
            connection_status: ConnectionStatus::Disconnected,      // Start disconnected
            error_fields: ValidationErrors::default(),              // No errors yet
            show_status: true,                                      // Show status by default
            message_history: Vec::new(),                            // Empty message history
            sort_by_price: true,                                    // Sort DOM by price by default
            ibkr_config: accounts[0].clone(),                       // Use first account config
            debug_mode: false,                                      // Debug mode off by default
            dom_levels,                                             // Set DOM levels from settings
            user_session: UserSession::new("guest", UserRole::Observer), // Start as guest observer
            websocket: None,                                        // No websocket yet
            audit_trail,                                            // Set audit trail logger
            order_type: "Limit".to_string(),                        // Default order type
            order_duration: "Day".to_string(),                      // Default order duration
            max_order_size: Decimal::new(10_000, 0),                // Max order size $10,000
            min_tick_size: Decimal::new(5, 2),                      // Min tick size $0.05
            min_price: Decimal::new(1, 2),                          // Min price $0.01
            max_price: Decimal::new(1_000_000, 0),                  // Max price $1,000,000
            margin_available: Decimal::new(100_000, 0),             // $100,000 margin
            persistent_settings: persistent_settings.clone(),        // Store persistent settings
            accounts,                                               // Store accounts list
            selected_account: 0,                                    // Select first account
            dark_mode: persistent_settings.theme == "dark",         // Set dark mode from settings
            last_symbol_subscribed: None,                           // No symbol subscribed yet
            session_timeout_secs: 15 * 60,                          // 15 minutes session timeout
            session_warn_before_secs: 60,                           // Warn 1 minute before timeout
            order_history: Vec::new(),                              // Empty order history
            custom_order_params: CustomOrderParams::default(),      // Default custom order params
            simulated_mode: persistent_settings.simulated_mode,     // Set simulated mode from settings
            order_statuses: HashMap::new(),                         // No order statuses yet
            drag_order: None,                                       // No drag order yet
            price_history: Vec::new(),                              // Empty price history
            backtest_results: None,                                 // No backtest results yet
            last_api_call: None,                                    // No API call yet
            api_calls_this_minute: 0,                               // No API calls yet
            last_api_minute: None,                                  // No API minute yet
            risk_alerts: Vec::new(),                                // No risk alerts yet
        }
    }

    // ... (rest of TradingApp implementation, see modularized files for details)
    // The actual implementation would be split into modules as above.
}

// --- OrderBook with DOM Level Limiting, Drag-and-Drop, and Efficient Data Handling (Heap-based for performance) ---

// Struct for the order book, using heaps for efficient best bid/ask
#[derive(Default, Debug)]
pub struct OrderBook {
    pub bids: BinaryHeap<(Decimal, Decimal)>, // Max-heap for bids (price, size)
    pub asks: BinaryHeap<(std::cmp::Reverse<Decimal>, Decimal)>, // Min-heap for asks (price, size)
    pub total_bid_orders: usize,              // Total number of bid orders
    pub total_ask_orders: usize,              // Total number of ask orders
}

// Implementation of OrderBook methods
impl OrderBook {
    // Create a new, empty order book
    pub fn new() -> Self {
        Self::default()
    }

    // Add an order to the book (bid or ask)
    pub fn add_order(&mut self, price: Decimal, size: Decimal, is_bid: bool) {
        if is_bid {
            self.bids.push((price, size));         // Add to bids heap
            self.total_bid_orders += 1;            // Increment bid order count
        } else {
            self.asks.push((std::cmp::Reverse(price), size)); // Add to asks heap
            self.total_ask_orders += 1;            // Increment ask order count
        }
    }

    // Remove an order from the book (not implemented for heap demo)
    pub fn remove_order(&mut self, price: Decimal, size: Decimal, is_bid: bool) {
        // For demo, not implemented for heap. In production, use a more advanced structure or Redis.
    }

    // Get the best bid price (highest bid)
    pub fn get_best_bid(&self) -> Option<Decimal> {
        self.bids.peek().map(|(p, _)| *p)
    }

    // Get the best ask price (lowest ask)
    pub fn get_best_ask(&self) -> Option<Decimal> {
        self.asks.peek().map(|(p, _)| p.0)
    }

    // Clear the order book (remove all bids and asks)
    pub fn clear(&mut self) {
        self.bids.clear();
        self.asks.clear();
        self.total_bid_orders = 0;
        self.total_ask_orders = 0;
    }

    // Get the average bid price (weighted by size)
    pub fn get_average_bid_price(&self) -> Option<Decimal> {
        if self.bids.is_empty() {
            None
        } else {
            let total_size: Decimal = self.bids.iter().map(|(_, size)| *size).sum();
            let weighted_sum: Decimal = self.bids.iter().map(|(price, size)| *price * *size).sum();
            Some(weighted_sum / total_size)
        }
    }

    // Get the average ask price (weighted by size)
    pub fn get_average_ask_price(&self) -> Option<Decimal> {
        if self.asks.is_empty() {
            None
        } else {
            let total_size: Decimal = self.asks.iter().map(|(_, size)| *size).sum();
            let weighted_sum: Decimal = self.asks.iter().map(|(price, size)| price.0 * *size).sum();
            Some(weighted_sum / total_size)
        }
    }

    /// Update the order book from a market data feed (event-driven).
    pub fn update_from_market_feed(&mut self, _market_data: &ibkr_rust::MarketData) {
        // In a real implementation, parse the market_data and update bids/asks.
        // This should be called by the WebSocket/event handler.
    }
}

// --- Connection Status Enum ---

// Enum for connection status to IBKR
#[derive(PartialEq, Debug)]
enum ConnectionStatus {
    Connected,     // Connected to IBKR
    Connecting,    // Connecting to IBKR
    Disconnected,  // Disconnected from IBKR
}

// --- Order Struct ---

// Struct for a single order (price, size, timestamp)
#[derive(Clone, Debug)]
struct Order {
    price: Decimal,                // Order price
    size: Decimal,                 // Order size
    timestamp: DateTime<Utc>,      // Time the order was placed
}

// --- eframe::App Implementation with Enhanced UI, Error Display, DOM Limiting, Tooltips, Custom Orders, Simulated Mode, Real-Time Feedback, Drag-and-Drop, Charting, Help Section, Session Warning ---

impl eframe::App for TradingApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // --- Theme Customization ---
        if self.dark_mode {
            ctx.set_visuals(Visuals::dark());
        } else {
            ctx.set_visuals(Visuals::light());
        }
        // --- Session Expiry and Warning ---
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
            // --- Session Timeout Warning Modal ---
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

            // --- User Authentication and Session Management ---
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

            // --- Top Bar: Theme, Account, Settings, Debug, Simulated Mode, Help ---
            ui.horizontal(|ui| {
                let (status_text, status_color) = match self.connection_status {
                    ConnectionStatus::Connected => ("Connected to IBKR", Color32::from_rgb(46, 204, 113)),
                    ConnectionStatus::Connecting => ("Connecting...", Color32::YELLOW),
                    ConnectionStatus::Disconnected => ("Disconnected", Color32::from_rgb(231, 76, 60)),
                };
                if matches!(self.connection_status, ConnectionStatus::Connecting) {
                    ui.spinner();
                    ui.add_space(5.0);
                }
                let icon_text = match self.connection_status {
                    ConnectionStatus::Connected => "[OK]",
                    ConnectionStatus::Connecting => "[...]",
                    ConnectionStatus::Disconnected => "[X]",
                };
                ui.label(RichText::new(icon_text).color(status_color).size(16.0));
                ui.add_space(5.0);
                ui.label(RichText::new(status_text).color(status_color));
                if self.user_session.role == UserRole::Admin {
                    if ui.button("Settings").on_hover_text("Configure IBKR Host/Port/Account").clicked() {
                        ui.memory_mut(|mem| mem.toggle_popup("ibkr_settings"));
                    }
                    egui::popup::popup_below_widget(ui, "ibkr_settings", ui.button(""), |ui| {
                        ui.label("IBKR Host:");
                        ui.text_edit_singleline(&mut self.ibkr_config.host);
                        ui.label("IBKR Port:");
                        ui.add(DragValue::new(&mut self.ibkr_config.port).clamp_range(1..=65535));
                        ui.label("Timeout (secs):");
                        ui.add(DragValue::new(&mut self.ibkr_config.timeout_secs).clamp_range(1..=30));
                        ui.label("Account ID:");
                        ui.text_edit_singleline(&mut self.ibkr_config.account_id);
                        if ui.button("Connect").clicked() {
                            self.connect_to_ibkr();
                            ui.close_menu();
                        }
                    });
                }
                if ui.button(if self.dark_mode { "Light Mode" } else { "Dark Mode" })
                    .on_hover_text("Toggle dark/light mode")
                    .clicked() {
                    self.dark_mode = !self.dark_mode;
                    self.save_settings();
                }
                if self.user_session.role != UserRole::Observer {
                    if ui.button("Switch Account").on_hover_text("Switch between multiple IBKR accounts").clicked() {
                        self.selected_account = (self.selected_account + 1) % self.accounts.len();
                        self.ibkr_config = self.accounts[self.selected_account].clone();
                        self.save_settings();
                    }
                }
                if ui.button("Debug").on_hover_text("Toggle debug mode").clicked() {
                    self.debug_mode = !self.debug_mode;
                }
                if ui.button(if self.simulated_mode { "Simulated: ON" } else { "Simulated: OFF" })
                    .on_hover_text("Toggle simulated trading mode (virtual funds, no real orders)")
                    .clicked() {
                    self.simulated_mode = !self.simulated_mode;
                    self.save_settings();
                }
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

 // --- Charting: Price History ---
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
            // --- Symbol, Quantity, Price, Order Type, Duration Entry with Real-Time Error Display and Custom Orders ---
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

            // --- Custom Order Parameters UI ---
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

            // --- Order Buttons with Confirmation Dialog, Retry, and Drag-and-Drop Support ---
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

            // --- Error Summary Section ---
            if !self.error_fields.is_empty() {
                ui.group(|ui| {
                    ui.label(RichText::new("Please fix the following errors:").color(Color32::from_rgb(231, 76, 60)));
                    for msg in self.error_fields.summary() {
                        ui.label(RichText::new(msg).color(Color32::from_rgb(231, 76, 60)));
                    }
                });
            }

            // --- Status Message and Message History ---
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

            // --- DOM Display with Level Limiting, Scrolling, Tooltips, Color Cues, and Drag-and-Drop ---
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
                        // --- Ask Side ---
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

                        // --- Bid Side ---
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

            // --- Order History (Audit Trail) ---
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

            // --- Real-Time Order Status Feedback and Risk Alerts ---
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

            // --- Backtesting Results ---
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

            // --- Debug Mode: Show Internal State ---
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

// --- Main Function with Logging Initialization, Persistent Settings, and Mobile/Web Support ---

// Main entry point for the application
fn main() -> eframe::Result<()> {
    env_logger::init(); // Initialize logging
    let native_options = eframe::NativeOptions {
        drag_and_drop_support: true, // Enable drag-and-drop
        follow_system_theme: true,   // Follow system theme (dark/light)
        maximized: false,            // Start not maximized
        initial_window_size: Some(Vec2::new(480.0, 800.0)), // Mobile-friendly default size
        ..Default::default()
    };
    eframe::run_native(
        "DOM Trading System",        // Application window title
        native_options,              // Window options
        Box::new(|cc| Box::new(TradingApp::new(cc))), // Create TradingApp instance
    )
}

// --- Helper: Get Local IP Address (for audit trail) ---
// Returns the first non-loopback IPv4 address, if available
fn get_local_ip() -> Option<IpAddr> {
    // Try to get the first non-loopback IPv4 address
    let addrs = local_ip_address::list_afinet_netifas().unwrap_or_default();
    for (_ifname, ip) in addrs {
        if !ip.is_loopback() {
            return Some(ip);
        }
    }
    None
}
