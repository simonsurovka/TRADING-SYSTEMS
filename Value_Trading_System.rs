// Imports and Dependencies
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc, Local};
use log::{info, warn, error};
use polars::prelude::{DataFrame, Series, CsvReader, CsvWriter, AnyValue};
use std::error::Error;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::Path;
use thiserror::Error;
use env_logger;
use serde_json::to_writer_pretty;

// Data Structures

// Market data for a single asset
#[derive(Debug, Serialize, Deserialize, Default, Clone)]
pub struct MarketData {
    pub symbol: String,
    pub price: f64,
    pub volume: f64,
    pub free_cash_flow: f64,
    pub current_ratio: f64,
    pub quick_ratio: f64,
    pub gross_margin: f64,
    pub operating_margin: f64,
    pub net_profit_margin: f64,
    pub beta: f64,
    pub shares_outstanding: u64,
    pub operating_cash_flow: f64,
    pub book_to_market_ratio: f64,
    pub dividend_yield: f64,
    pub debt_to_equity: f64,
    pub return_on_equity: f64,
    pub book_value_per_share: f64,
    pub market_cap: f64,
    pub price_to_book: f64,
    pub revenue_per_share: f64,
    pub fundamental_data: HashMap<String, f64>,
    pub revenue_growth: f64,
    pub timestamp: String,
    pub pe_ratio: f64,
}

// Trade signal 
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TradeSignal {
    pub asset: String,
    pub action: String,
    pub quantity: f64,
    pub price: f64,
    pub reason: String,
    pub timestamp: DateTime<Utc>,
    pub risk_metrics: RiskMetrics,
    pub analysis: HashMap<String, f64>,
}

// Risk metrics
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RiskMetrics {
    pub volatility: f64,
    pub max_drawdown: f64,
    pub sharpe_ratio: f64,
    pub position_size_pct: f64,
    pub portfolio_drawdown: f64,
}

// Portfolio metrics
#[derive(Debug, Clone)]
pub struct PortfolioMetrics {
    pub total_value: f64,
    pub cash_balance: f64,
    pub equity_value: f64,
    pub positions: HashMap<String, Position>,
    pub diversification_score: f64,
    pub herfindahl_index: f64,
    pub peak_equity: f64,
}

// Single asset position in the portfolio
#[derive(Debug, Clone)]
pub struct Position {
    pub quantity: f64,
    pub avg_price: f64,
    pub current_price: f64,
    pub market_value: f64,
    pub unrealized_pnl: f64,
    pub weight: f64,
    pub stop_loss: Option<f64>,
    pub take_profit: Option<f64>,
    pub last_rebalance: DateTime<Utc>,
}

// Helper Macros
macro_rules! extract_field {
    ($df:expr, $col:expr, $idx:expr, $field:expr) => {
        match $df.column($col)?.get($idx) {
            Some(AnyValue::Float64(val)) => Ok(val),
            Some(AnyValue::Int64(val)) => Ok(val as f64),
            Some(AnyValue::UInt64(val)) => Ok(val as f64),
            Some(AnyValue::UInt32(val)) => Ok(val as f64),
            Some(AnyValue::Int32(val)) => Ok(val as f64),
            Some(AnyValue::Float32(val)) => Ok(val as f64),
            _ => {
                warn!("Failed to extract {} at index {}", $col, $idx);
                Err(Box::<dyn Error>::from(format!("Invalid {} data", $field)))
            }
        }
    };
}

// Error Handling
#[derive(Error, Debug)]
pub enum TradingError {
    #[error("Invalid market data: {0}")]
    InvalidMarketData(String),
    #[error("Data conversion error: {0}")]
    DataConversion(String),
    #[error("Risk limit exceeded: {0}")]
    RiskLimitExceeded(String),
}

// Strategy Configuration
pub struct StrategyConfig {
    pub pe_ratio_threshold: f64,
    pub price_to_book_threshold: f64,
    pub free_cash_flow_threshold: f64,
    pub current_ratio_threshold: f64,
    pub quick_ratio_threshold: f64,
    pub gross_margin_threshold: f64,
    pub operating_margin_threshold: f64,
    pub net_profit_margin_threshold: f64,
    pub debt_to_equity_threshold: f64,
    pub max_position_size: f64,
    pub volatility_threshold: f64,
    pub max_drawdown_limit: f64,
    pub rolling_window_size: usize,
    pub position_drift_threshold: f64,
}

// Strategy Trait

pub trait Strategy {
    fn generate_signal(&mut self, data: &MarketData) -> Result<TradeSignal, TradingError>;
}

// Market Data Strategy Implementation 
pub struct MarketDataStrategy {
    config: StrategyConfig,
    rolling_max_price: f64,
    returns: Vec<f64>,
    prices: Vec<f64>,
    peak_portfolio_value: f64,
}
impl MarketDataStrategy {
    pub fn new(config: StrategyConfig) -> Self {
        Self {
            config,
            rolling_max_price: 0.0,
            returns: Vec::with_capacity(config.rolling_window_size),
            prices: Vec::with_capacity(config.rolling_window_size),
            peak_portfolio_value: 0.0,
        }
    }

    // Check value metrics against thresholds
    fn check_value_metrics(&self, data: &MarketData) -> Result<bool, TradingError> {
        if data.pe_ratio <= 0.0 || data.price_to_book <= 0.0 {
            return Err(TradingError::InvalidMarketData("Invalid value metrics".into()));
        }
        Ok(data.pe_ratio < self.config.pe_ratio_threshold 
            && data.price_to_book < self.config.price_to_book_threshold)
    }

    // Check financial health metrics against thresholds
    fn check_financial_health(&self, data: &MarketData) -> Result<bool, TradingError> {
        if data.free_cash_flow <= 0.0 || data.current_ratio <= 0.0 || data.quick_ratio <= 0.0 {
            return Err(TradingError::InvalidMarketData("Invalid financial health metrics".into()));
        }
        Ok(data.free_cash_flow > self.config.free_cash_flow_threshold
            && data.current_ratio > self.config.current_ratio_threshold
            && data.quick_ratio > self.config.quick_ratio_threshold
            && data.gross_margin > self.config.gross_margin_threshold
            && data.dividend_yield > 0.0)
    }

    // Risk metrics calculation
    fn calculate_risk_metrics(&mut self, data: &MarketData, portfolio_value: f64) -> RiskMetrics {
        self.peak_portfolio_value = self.peak_portfolio_value.max(portfolio_value);
        let portfolio_drawdown = if portfolio_value < self.peak_portfolio_value {
            (self.peak_portfolio_value - portfolio_value) / self.peak_portfolio_value
        } else {
            0.0
        };

        if self.prices.len() >= self.config.rolling_window_size {
            self.prices.remove(0);
        }
        self.prices.push(data.price);

        if self.prices.len() > 1 {
            let last_price = self.prices[self.prices.len() - 2];
            let return_pct = (data.price - last_price) / last_price;
            self.returns.push(return_pct);
        } else {
            self.returns.push(0.0);
        }

        let avg_return = self.returns.iter().sum::<f64>() / self.returns.len() as f64;
        let return_variance = self.returns.iter()
            .map(|r| (r - avg_return).powi(2))
            .sum::<f64>() / self.returns.len() as f64;
        let volatility = return_variance.sqrt();
        self.rolling_max_price = self.rolling_max_price.max(data.price);
        let max_drawdown = if data.price < self.rolling_max_price {
            (self.rolling_max_price - data.price) / self.rolling_max_price
        } else {
            0.0
        };

        let risk_free_rate = 0.02;
        let excess_return = avg_return - risk_free_rate;
        let sharpe_ratio = if self.returns.len() > 1 && return_variance > 0.0 {
            excess_return / return_variance.sqrt()
        } else {
            0.0
        };

        RiskMetrics {
            volatility,
            max_drawdown,
            sharpe_ratio,
            position_size_pct: 0.0,
            portfolio_drawdown,
        }
    }

    // Trade quantity based on risk and position sizing calculation
    fn calculate_trade_quantity(&self, data: &MarketData, risk_metrics: &mut RiskMetrics) -> f64 {
        let base_position = 10000.0;

        let vol_factor = (1.0 - risk_metrics.volatility).max(0.2);
        let size_factor = (data.market_cap / 1_000_000_000.0).min(2.0).max(0.5);
        let risk_factor = if data.beta > self.config.volatility_threshold { 0.5 } else { 1.0 };
        let margin_factor = if data.operating_margin > self.config.operating_margin_threshold { 1.2 } else { 1.0 };

        let max_quantity = (self.config.max_position_size * base_position) / data.price;
        let position_size = base_position * vol_factor * size_factor * risk_factor * margin_factor;
        let quantity = (position_size / data.price).min(max_quantity);
        risk_metrics.position_size_pct = quantity * data.price / base_position;
        quantity
    }
}

// Strategy trait implementation for MarketDataStrategy
impl Strategy for MarketDataStrategy {
    fn generate_signal(&mut self, data: &MarketData) -> Result<TradeSignal, TradingError> {
        let mut risk_metrics = self.calculate_risk_metrics(data, self.peak_portfolio_value);
        let mut analysis = HashMap::new();

        analysis.insert("volatility".to_string(), risk_metrics.volatility);
        analysis.insert("max_drawdown".to_string(), risk_metrics.max_drawdown);
        analysis.insert("pe_ratio".to_string(), data.pe_ratio);
        analysis.insert("price_to_book".to_string(), data.price_to_book);

        if risk_metrics.max_drawdown > self.config.max_drawdown_limit {
            return Err(TradingError::RiskLimitExceeded(
                format!("Max drawdown {} exceeds limit {}", 
                    risk_metrics.max_drawdown, self.config.max_drawdown_limit)
            ));
        }

        let quantity = self.calculate_trade_quantity(data, &mut risk_metrics);
        let timestamp = Utc::now();

        let signal = if self.check_value_metrics(data)? && self.check_financial_health(data)? {
            TradeSignal {
                action: "buy".to_string(),
                quantity,
                price: data.price,
                asset: data.symbol.clone(),
                reason: "Value metrics and financial health criteria met".to_string(),
                timestamp,
                risk_metrics,
                analysis,
            }
        } else if data.pe_ratio > 25.0 || data.debt_to_equity > self.config.debt_to_equity_threshold {
            TradeSignal {
                action: "sell".to_string(),
                quantity,
                price: data.price,
                asset: data.symbol.clone(),
                reason: "Risk metrics exceeded thresholds".to_string(),
                timestamp,
                risk_metrics,
                analysis,
            }
        } else {
            TradeSignal {
                action: "hold".to_string(),
                quantity: 0.0,
                price: data.price,
                asset: data.symbol.clone(),
                reason: "No clear signal".to_string(),
                timestamp,
                risk_metrics,
                analysis,
            }
        };

        info!("Generated signal: {:?}", signal);
        Ok(signal)
    }
}

// Backtesting Engine

// Simulates trading over historical data
pub struct BacktestingEngine {
    historical_data: DataFrame,
    strategy: Box<dyn Strategy>,
    trades: Vec<TradeSignal>,
    portfolio_value: f64,
    available_funds: f64,
    holdings: HashMap<String, Position>,
}

impl BacktestingEngine {
    pub fn new(historical_data: DataFrame, strategy: Box<dyn Strategy>) -> Self {
        Self {
            historical_data,
            strategy,
            trades: Vec::new(),
            portfolio_value: 100000.0,
            available_funds: 100000.0,
            holdings: HashMap::new(),
        }
    }

    // Backtest 
    pub fn run(&mut self) -> Result<(), Box<dyn Error>> {
        let row_count = self.historical_data.height();
        for i in 0..row_count {
            let market_data = self.convert_row_to_market_data(i)?;
            match self.strategy.generate_signal(&market_data) {
                Ok(signal) => {
                    self.execute_trade(&signal, &market_data);
                    self.portfolio_value = self.calculate_portfolio_value();
                },
                Err(e) => warn!("Failed to generate signal: {}", e),
            }
        }

        let file = File::create("trades.json")?;
        to_writer_pretty(file, &self.trades)?;

        Ok(())
    }

    // Convert a DataFrame row into a MarketData struct
    fn convert_row_to_market_data(&self, index: usize) -> Result<MarketData, Box<dyn Error>> {
        let mut market_data = MarketData::default();

        market_data.symbol = match self.historical_data.column("symbol")?.get(index) {
            Some(AnyValue::String(s)) => s.to_string(),
            _ => return Err(Box::<dyn Error>::from("Invalid symbol data"))
        };
        market_data.price = extract_field!(self.historical_data, "price", index, "price")?;
        market_data.volume = extract_field!(self.historical_data, "volume", index, "volume")?;
        market_data.free_cash_flow = extract_field!(self.historical_data, "free_cash_flow", index, "free cash flow")?;
        market_data.current_ratio = extract_field!(self.historical_data, "current_ratio", index, "current ratio")?;
        market_data.quick_ratio = extract_field!(self.historical_data, "quick_ratio", index, "quick ratio")?;
        market_data.gross_margin = extract_field!(self.historical_data, "gross_margin", index, "gross margin")?;
        market_data.operating_margin = extract_field!(self.historical_data, "operating_margin", index, "operating margin")?;
        market_data.net_profit_margin = extract_field!(self.historical_data, "net_profit_margin", index, "net profit margin")?;
        market_data.beta = extract_field!(self.historical_data, "beta", index, "beta")?;
        market_data.shares_outstanding = extract_field!(self.historical_data, "shares_outstanding", index, "shares outstanding")? as u64;
        market_data.operating_cash_flow = extract_field!(self.historical_data, "operating_cash_flow", index, "operating cash flow")?;
        market_data.book_to_market_ratio = extract_field!(self.historical_data, "book_to_market_ratio", index, "book to market ratio")?;
        market_data.dividend_yield = extract_field!(self.historical_data, "dividend_yield", index, "dividend yield")?;
        market_data.debt_to_equity = extract_field!(self.historical_data, "debt_to_equity", index, "debt to equity")?;
        market_data.return_on_equity = extract_field!(self.historical_data, "return_on_equity", index, "return on equity")?;
        market_data.book_value_per_share = extract_field!(self.historical_data, "book_value_per_share", index, "book value per share")?;
        market_data.market_cap = extract_field!(self.historical_data, "market_cap", index, "market cap")?;
        market_data.price_to_book = extract_field!(self.historical_data, "price_to_book", index, "price to book")?;
        market_data.revenue_per_share = extract_field!(self.historical_data, "revenue_per_share", index, "revenue per share")?;
        market_data.revenue_growth = extract_field!(self.historical_data, "revenue_growth", index, "revenue growth")?;
        market_data.timestamp = match self.historical_data.column("timestamp")?.get(index) {
            Some(AnyValue::String(s)) => s.to_string(),
            _ => "".to_string()
        };
        market_data.pe_ratio = match extract_field!(self.historical_data, "pe_ratio", index, "PE ratio") {
            Ok(pe) => pe,
            Err(_) => {
                warn!("PE ratio missing at index {}, using default", index);
                0.0
            }
        };

        Ok(market_data)
    }

    // Execute a trade signal and update holdings and cash
    fn execute_trade(&mut self, signal: &TradeSignal, _market_data: &MarketData) {
        info!("Executing trade: {:?}", signal);
        match signal.action.as_str() {
            "buy" => {
                let cost = signal.quantity * signal.price;
                if cost <= self.available_funds && signal.quantity > 0.0 {
                    self.available_funds -= cost;
                    let position = self.holdings.entry(signal.asset.clone())
                        .or_insert(Position {
                            quantity: 0.0,
                            avg_price: 0.0,
                            current_price: signal.price,
                            market_value: 0.0,
                            unrealized_pnl: 0.0,
                            weight: 0.0,
                            stop_loss: Some(signal.price * (1.0 - signal.risk_metrics.volatility)),
                            take_profit: Some(signal.price * (1.0 + signal.risk_metrics.volatility * 2.0)),
                            last_rebalance: Utc::now(),
                        });
                    let total_cost = position.quantity * position.avg_price + cost;
                    let total_quantity = position.quantity + signal.quantity;
                    position.avg_price = if total_quantity > 0.0 { total_cost / total_quantity } else { 0.0 };
                    position.quantity = total_quantity;
                    position.current_price = signal.price;
                    position.market_value = total_quantity * signal.price;
                    position.unrealized_pnl = (signal.price - position.avg_price) * total_quantity;
                }
            }
            "sell" => {
                if let Some(position) = self.holdings.get_mut(&signal.asset) {
                    if position.quantity >= signal.quantity && signal.quantity > 0.0 {
                        position.quantity -= signal.quantity;
                        let proceeds = signal.quantity * signal.price;
                        self.available_funds += proceeds;
                        position.current_price = signal.price;
                        position.market_value = position.quantity * signal.price;
                        position.unrealized_pnl = (signal.price - position.avg_price) * position.quantity;
                        if position.quantity == 0.0 {
                            self.holdings.remove(&signal.asset);
                        }
                    }
                }
            }
            "hold" => {
            }
            _ => {}
        }
        self.trades.push(signal.clone());
    }

    // Calculate the total portfolio value 
    fn calculate_portfolio_value(&self) -> f64 {
        let mut total_value = self.available_funds;
        for position in self.holdings.values() {
            total_value += position.quantity * position.current_price;
        }
        total_value
    }
}

// Execution Engine

// Handles live trade execution and portfolio tracking
pub struct ExecutionEngine {
    available_funds: f64,
    holdings: HashMap<String, Position>,
    metrics: PortfolioMetrics,
}

impl ExecutionEngine {
    pub fn new(initial_funds: f64) -> Self {
        Self {
            available_funds: initial_funds,
            holdings: HashMap::new(),
            metrics: PortfolioMetrics {
                total_value: initial_funds,
                cash_balance: initial_funds,
                equity_value: 0.0,
                positions: HashMap::new(),
                diversification_score: 0.0,
                herfindahl_index: 0.0,
                peak_equity: initial_funds,
            },
        }
    }

    // Execute a trade signal and update portfolio metrics and logs
    pub fn execute_trade(&mut self, signal: &TradeSignal) -> Result<(), Box<dyn Error>> {
        let execution_status = match signal.action.as_str() {
            "buy" => self.execute_buy(signal),
            "sell" => self.execute_sell(signal),
            "hold" => self.handle_hold(signal),
            _ => Err(Box::<dyn Error>::from("Invalid action"))
        };

        match execution_status {
            Ok(status) => {
                self.update_portfolio_metrics(signal)?;
                self.track_execution_status(&status, signal)?;
                Ok(())
            }
            Err(e) => {
                error!("Trade execution failed: {}", e);
                Err(Box::<dyn Error>::from(e))
            }
        }
    }

    // Handle a hold signal, possibly rebalancing the position
    fn handle_hold(&mut self, signal: &TradeSignal) -> Result<String, Box<dyn Error>> {
        if let Some(position) = self.holdings.get_mut(&signal.asset) {
            let new_stop_loss = signal.price * (1.0 - signal.risk_metrics.volatility * 1.5);
            position.stop_loss = Some(new_stop_loss.max(position.stop_loss.unwrap_or(0.0)));
            if signal.risk_metrics.volatility > 0.0 {
                position.stop_loss = Some(signal.price * (1.0 - signal.risk_metrics.volatility));
            }

            let target_weight = 1.0 / self.holdings.len() as f64;
            let drift = (position.weight - target_weight).abs();

            if drift > 0.05 {
                let required_adjustment = (target_weight - position.weight) * self.metrics.total_value;
                let adjustment_quantity = required_adjustment / signal.price;

                if adjustment_quantity > 0.0 && self.available_funds >= required_adjustment {
                    position.quantity += adjustment_quantity;
                    self.available_funds -= required_adjustment;
                    return Ok(format!("Rebalanced position in {} by adding {} units", 
                        signal.asset, adjustment_quantity));
                } else if adjustment_quantity < 0.0 {
                    position.quantity += adjustment_quantity;
                    self.available_funds -= required_adjustment;
                    return Ok(format!("Rebalanced position in {} by removing {} units", 
                        signal.asset, adjustment_quantity.abs()));
                }
            }
        }

        Ok(format!("Holding position in {}. No rebalancing needed.", signal.asset))
    }

    // Execute and update the position
    fn execute_buy(&mut self, signal: &TradeSignal) -> Result<String, Box<dyn Error>> {
        let cost = signal.quantity * signal.price;

        if cost > self.available_funds {
            return Err(Box::<dyn Error>::from("Insufficient funds for purchase"));
        }

        self.available_funds -= cost;

        let position = self.holdings.entry(signal.asset.clone())
            .or_insert(Position {
                quantity: 0.0,
                avg_price: 0.0,
                current_price: signal.price,
                market_value: 0.0,
                unrealized_pnl: 0.0,
                weight: 0.0,
                stop_loss: Some(signal.price * (1.0 - signal.risk_metrics.volatility)),
                take_profit: Some(signal.price * (1.0 + signal.risk_metrics.volatility * 2.0)),
                last_rebalance: Utc::now(),
            });

        let total_cost = position.quantity * position.avg_price + cost;
        let total_quantity = position.quantity + signal.quantity;
        position.avg_price = if total_quantity > 0.0 { total_cost / total_quantity } else { 0.0 };
        position.quantity = total_quantity;
        position.market_value = total_quantity * signal.price;
        position.unrealized_pnl = (signal.price - position.avg_price) * total_quantity;
        position.current_price = signal.price;

        self.metrics.equity_value = self.holdings.values().map(|p| p.market_value).sum();
        position.weight = if self.metrics.total_value > 0.0 { position.market_value / self.metrics.total_value } else { 0.0 };

        Ok(format!("Bought {} units of {} at ${:.2} per unit", 
            signal.quantity, signal.asset, signal.price))
    }

    // Execute a sell signal and update/remove the position
    fn execute_sell(&mut self, signal: &TradeSignal) -> Result<String, Box<dyn Error>> {
        let position = self.holdings.get_mut(&signal.asset)
            .ok_or_else(|| Box::<dyn Error>::from("No position found for asset"))?;

        if position.quantity < signal.quantity {
            return Err(Box::<dyn Error>::from("Insufficient quantity to sell"));
        }

        position.quantity -= signal.quantity;
        let proceeds = signal.quantity * signal.price;
        self.available_funds += proceeds;

        if position.quantity == 0.0 {
            self.holdings.remove(&signal.asset);
        } else {
            position.market_value = position.quantity * signal.price;
            position.unrealized_pnl = (signal.price - position.avg_price) * position.quantity;
            position.weight = if self.metrics.total_value > 0.0 { position.market_value / self.metrics.total_value } else { 0.0 };
            position.current_price = signal.price;
        }

        self.metrics.equity_value = self.holdings.values().map(|p| p.market_value).sum();

        Ok(format!("Sold {} units of {} for ${:.2}", 
            signal.quantity, signal.asset, proceeds))
    }

    // Update portfolio metrics after a trade
    fn update_portfolio_metrics(&mut self, _signal: &TradeSignal) -> Result<(), Box<dyn Error>> {
        let total_positions = self.holdings.len() as f64;

        if total_positions == 0.0 {
            self.metrics.diversification_score = 0.0;
            self.metrics.herfindahl_index = 1.0;
        } else {
            let weights_squared_sum: f64 = self.holdings.values()
                .map(|p| p.weight.powi(2))
                .sum();
            self.metrics.herfindahl_index = weights_squared_sum;
            self.metrics.diversification_score = 1.0 - weights_squared_sum;
        }
        self.metrics.equity_value = self.holdings.values().map(|p| p.market_value).sum();
        self.metrics.total_value = self.available_funds + self.metrics.equity_value;
        self.metrics.peak_equity = self.metrics.peak_equity.max(self.metrics.total_value);

        // The following methods are assumed to be implemented elsewhere in ExecutionEngine
        let portfolio_sharpe = self.calculate_portfolio_sharpe_ratio();
        let portfolio_sortino = self.calculate_portfolio_sortino_ratio();
        let max_drawdown = self.calculate_max_drawdown();
        let volatility = self.calculate_portfolio_volatility();

        info!(
            "Portfolio Metrics | Sharpe: {:.4}, Sortino: {:.4}, Max Drawdown: {:.2}%, Volatility: {:.4}",
            portfolio_sharpe,
            portfolio_sortino,
            max_drawdown * 100.0,
            volatility
        );
        info!("Updated portfolio metrics: {:#?}", self.metrics);

        if let Ok(mut file) = OpenOptions::new()
            .append(true)
            .create(true)
            .open("portfolio_metrics.csv")
        {
            if let Ok(metadata) = file.metadata() {
                if metadata.len() == 0 {
                    writeln!(
                        file,
                        "timestamp,sharpe,sortino,max_drawdown,volatility,total_value,peak_equity,cash_balance,equity_value"
                    )?;
                }
            } else {
                writeln!(
                    file,
                    "timestamp,sharpe,sortino,max_drawdown,volatility,total_value,peak_equity,cash_balance,equity_value"
                )?;
            }
            writeln!(
                file,
                "{},{:.4},{:.4},{:.4},{:.4},{:.2},{:.2},{:.2},{:.2}",
                Local::now().to_rfc3339(),
                portfolio_sharpe,
                portfolio_sortino,
                max_drawdown,
                volatility,
                self.metrics.total_value,
                self.metrics.peak_equity,
                self.available_funds,
                self.metrics.equity_value
            )?;
        }
        Ok(())
    }

    // Log the execution status of a trade
    fn track_execution_status(&self, status: &str, signal: &TradeSignal) -> Result<(), Box<dyn Error>> {
        let mut file = OpenOptions::new()
            .append(true)
            .create(true)
            .open("execution_status.log")?;

        let log_entry = format!(
            "[{}] {} - Action: {},
             Asset: {}, Quantity: {},
             Price: {:.2}, Reason: {}, 
             Risk Metrics: {:?},
             Analysis: {:?}",
            Local::now().to_rfc3339(),
            status,
            signal.action,
            signal.asset,
            signal.quantity,
            signal.price,
            signal.reason,
            signal.risk_metrics,
            signal.analysis
        );
        writeln!(file, "{}", log_entry)?;
        info!("{}", log_entry);

        Ok(())
    }
}

// Main Application Entry Point
fn main() -> Result<(), Box<dyn Error>> {
    env_logger::init();

    // Set up strategy configuration
    let config = StrategyConfig {
        pe_ratio_threshold: 15.0,
        price_to_book_threshold: 1.0,
        free_cash_flow_threshold: 0.0,
        current_ratio_threshold: 1.5,
        quick_ratio_threshold: 1.0,
        gross_margin_threshold: 0.30,
        operating_margin_threshold: 0.10,
        net_profit_margin_threshold: 0.10,
        debt_to_equity_threshold: 0.5,
        max_position_size: 0.10,
        volatility_threshold: 1.5,
        max_drawdown_limit: 0.2,
        rolling_window_size: 20,
        position_drift_threshold: 0.05,
    };

    // Load historical data from CSV
    let historical_data = CsvReader::from_path(Path::new("historical_data.csv"))?
        .finish()?;

    // Initialize strategy and backtesting engine
    let strategy = Box::new(MarketDataStrategy::new(config));
    let mut engine = BacktestingEngine::new(historical_data, strategy);
    engine.run()?;

    info!("Backtesting completed. Total trades: {}", engine.trades.len());
    Ok(())
}
