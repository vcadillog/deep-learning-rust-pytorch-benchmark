use rust::utils::learning_loop::LearningAlgorithm;
use rust::settings::Settings;

pub fn main(){
    let config = Settings::new().expect("loaded configuration");
    let _ = LearningAlgorithm::new(config).run();
}
