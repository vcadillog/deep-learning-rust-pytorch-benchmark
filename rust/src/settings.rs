use config::{Config, ConfigError, File};
use serde::Deserialize;

const CONFIG_FILE_PATH: &str = "../config.yaml";

#[derive(Debug, Deserialize, Clone)]
pub struct Settings{
    pub epochs: i32,
    pub shuffle: bool,
    pub image_dim: i64,
    pub labels: i64,
    pub units: i64,
    pub batch_size: i64,
    pub test_batch_size: i64,
    pub runs: i32,
    pub learning_rate: f64,
    pub layers_cpu: Vec<u32>,
    pub layers_cuda: Vec<u32>,
    pub use_cuda: Vec<bool>,
    pub log_dir: String,
    pub data_path: String,
}

impl Settings{
    pub fn new() -> Result<Self, ConfigError>{
        let settings = Config::builder()
                .add_source(File::with_name(CONFIG_FILE_PATH))
                .build()
                .unwrap();
        let settings: Settings = settings.try_deserialize().unwrap();
        return Ok(settings);
    }
}

