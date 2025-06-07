// Imports and Dependencies
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc, Local};
use log::{info, warn, error};
use polars::prelude::*;
use std::error::Error;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::Path;
use thiserror::Error;
use env_logger;
use tokio;

#[derive(Debug, Serialize, Deserialize, Default)]
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

// TradeSignal
#[derive(Debug, Serialize, Deserialize)]
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

// RiskMetrics
#[derive(Debug, Serialize, Deserialize)]
pub struct RiskMetrics {
    pub volatility: f64,
    pub max_drawdown: f64,
    pub sharpe_ratio: f64,
    pub position_size_pct: f64,
    pub portfolio_drawdown: f64,
}

// PortfolioMetrics
#[derive(Debug)]
pub struct PortfolioMetrics {
    pub total_value: f64,
    pub cash_balance: f64,
    pub equity_value: f64,
    pub positions: HashMap<String, Position>,
    pub diversification_score: f64,
    pub herfindahl_index: f64,
    pub peak_equity: f64,
}

// Position struct
#[derive(Debug)]
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
            _ => {
                warn!("Failed to extract {} at index {}", $col, $idx);
                Err(format!("Invalid {} data", $field).into())
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
            peak_portfolio_value: 0.0
        }
    }

    fn check_value_metrics(&self, data: &MarketData) -> Result<bool, TradingError> {
        if data.pe_ratio <= 0.0 || data.price_to_book <= 0.0 {
            return Err(TradingError::InvalidMarketData("Invalid value metrics".into()));
        }
        Ok(data.pe_ratio < self.config.pe_ratio_threshold 
            && data.price_to_book < self.config.price_to_book_threshold)
    }

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

    fn calculate_trade_quantity(&self, data: &MarketData, risk_metrics: &mut RiskMetrics) -> f64 {
        let base_position = 10000.0;
        
        let vol_factor = (1.0 - risk_metrics.volatility).max(0.2);
        let size_factor = (data.market_cap / 1_000_000_000.0).min(2.0).max(0.5);
        let risk_factor = if data.beta > self.config.volatility_threshold { 0.5 } else { 1.0 };
        let margin_factor = if data.operating_margin > self.config.operating_margin_threshold { 1.2 } else { 1.0 };
        
        let position_size = base_position * vol_factor * size_factor * risk_factor * margin_factor;
        let quantity = (position_size / data.price).min(self.config.max_position_size);
        risk_metrics.position_size_pct = quantity * data.price / base_position;
        quantity
    }
}

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
pub struct BacktestingEngine {
    historical_data: DataFrame,
    strategy: Box<dyn Strategy>,
    trades: Vec<TradeSignal>,
    portfolio_value: f64,
}

impl BacktestingEngine {
    pub fn new(historical_data: DataFrame, strategy: Box<dyn Strategy>) -> Self {
        Self {
            historical_data,
            strategy,
            trades: Vec::new(),
            portfolio_value: 100000.0,
        }
    }

    pub fn run(&mut self) -> Result<(), Box<dyn Error>> {
        let row_count = self.historical_data.height();
        for i in 0..row_count {
            let market_data = self.convert_row_to_market_data(i)?;
            match self.strategy.generate_signal(&market_data) {
                Ok(signal) => {
                    self.execute_trade(&signal);
                    self.portfolio_value = self.calculate_portfolio_value();
                },
                Err(e) => warn!("Failed to generate signal: {}", e),
            }
        }

        let file = File::create("trades.json")?;
        to_writer_pretty(file, &self.trades)?;

        Ok(())
    }

    fn convert_row_to_market_data(&self, index: usize) -> Result<MarketData, Box<dyn Error>> {
        let mut market_data = MarketData::default();
        
        market_data.symbol = match self.historical_data.column("symbol")?.get(index) {
            Some(AnyValue::String(s)) => s.to_string(),
            _ => return Err("Invalid symbol data".into())
        };

        market_data.price = extract_field!(self.historical_data, "price", index, "price")?;
        market_data.volume = extract_field!(self.historical_data, "volume", index, "volume")?;
        market_data.free_cash_flow = extract_field!(self.historical_data, "free_cash_flow", index, "free cash flow")?;
        market_data.current_ratio = extract_field!(self.historical_data, "current_ratio", index, "current ratio")?;
        market_data.quick_ratio = extract_field!(self.historical_data, "quick_ratio", index, "quick ratio")?;
        
        if let Ok(pe) = extract_field!(self.historical_data, "pe_ratio", index, "PE ratio") {
            market_data.pe_ratio = pe;
        } else {
            warn!("PE ratio missing at index {}, using default", index);
        }
        
        Ok(market_data)
    }

    fn execute_trade(&mut self, signal: &TradeSignal) {
        info!("Executing trade: {:?}", signal);
        self.trades.push(signal.clone());
    }

    fn calculate_portfolio_value(&self) -> f64 {
        let mut total_value = self.available_funds;

        for position in self.holdings.values() {
            total_value += position.quantity as f64 * position.current_price;
        }

        total_value
    }
}

// Execution Engine
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

    pub fn execute_trade(&mut self, signal: &TradeSignal) -> Result<(), Box<dyn Error>> {
        let execution_status = match signal.action.as_str() {
            "buy" => self.execute_buy(signal),
            "sell" => self.execute_sell(signal),
            "hold" => self.handle_hold(signal),
            _ => Err("Invalid action".into())
        };

        match execution_status {
            Ok(status) => {
                self.update_portfolio_metrics(signal)?;
                self.track_execution_status(&status, signal)?;
                Ok(())
            }
            Err(e) => {
                error!("Trade execution failed: {}", e);
                Err(e.into())
            }
        }
    }

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

    fn execute_buy(&mut self, signal: &TradeSignal) -> Result<String, Box<dyn Error>> {
        let cost = signal.quantity * signal.price;
        
        if cost > self.available_funds {
            return Err("Insufficient funds for purchase".into());
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
        position.avg_price = total_cost / total_quantity;
        position.quantity = total_quantity;
        position.market_value = total_quantity * signal.price;
        position.unrealized_pnl = (signal.price - position.avg_price) * total_quantity;
        
        self.metrics.equity_value += cost;
        position.weight = position.market_value / self.metrics.total_value;

        Ok(format!("Bought {} units of {} at ${:.2} per unit", 
            signal.quantity, signal.asset, signal.price))
    }

    fn execute_sell(&mut self, signal: &TradeSignal) -> Result<String, Box<dyn Error>> {
        let position = self.holdings.get_mut(&signal.asset)
            .ok_or_else(|| "No position found for asset".to_string())?;

        if position.quantity < signal.quantity {
            return Err("Insufficient quantity to sell".into());
        }

        position.quantity -= signal.quantity;
        let proceeds = signal.quantity * signal.price;
        self.available_funds += proceeds;
        self.metrics.equity_value -= proceeds;

        if position.quantity == 0.0 {
            self.holdings.remove(&signal.asset);
        } else {
            position.market_value = position.quantity * signal.price;
            position.unrealized_pnl = (signal.price - position.avg_price) * position.quantity;
            position.weight = position.market_value / self.metrics.total_value;
        }

        Ok(format!("Sold {} units of {} for ${:.2}", 
            signal.quantity, signal.asset, proceeds))
    }

    fn update_portfolio_metrics(&mut self, signal: &TradeSignal) -> Result<(), Box<dyn Error>> {
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
        
        self.metrics.total_value = self.available_funds + self.metrics.equity_value;
        self.metrics.peak_equity = self.metrics.peak_equity.max(self.metrics.total_value);
        
        info!("Updated portfolio metrics: {:?}", self.metrics);
        Ok(())
    }

    fn track_execution_status(&self, status: &str, signal: &TradeSignal) -> Result<(), Box<dyn Error>> {
        let mut file = OpenOptions::new()
            .append(true)
            .create(true)
            .open("execution_status.log")?;

        writeln!(file, "[{}] {} - Action: {}, Quantity: {}, Reason: {}, Risk Metrics: {:?}, Analysis: {:?}",
            Local::now(),
            status,
            signal.action,
            signal.quantity,
            signal.reason,
            signal.risk_metrics,
            signal.analysis
        )?;
        Ok(())
    }
}

// Main Application Entry Point
#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    env_logger::init();

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

    let historical_data = CsvReader::from_path(Path::new("historical_data.csv"))?
        .finish()?;

    let strategy = Box::new(MarketDataStrategy::new(config));
    let mut engine = BacktestingEngine::new(historical_data, strategy);
    
    engine.run()?;

    info!("Backtesting completed. Total trades: {}", engine.trades.len());
    
    Ok(())
}
