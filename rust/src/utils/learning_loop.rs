use anyhow::Result;
use tch::{nn,Device, nn::ModuleT, nn::OptimizerConfig, vision::dataset::Dataset,
Tensor, Kind, nn::Sequential};
use std::{time::Instant, cell::{Cell, RefCell}};

use crate::utils::network::network;
use crate::settings::Settings;
use crate::utils::logger::{LogConsole, LogCSV};

pub struct LearningAlgorithm{
    config: Settings,
    dataset: Dataset,
    device: Cell<Device>,
    model: RefCell<Sequential>,
}

impl LearningAlgorithm {    
    fn train_epoch(&self, hidden_layers: &u32) -> Result<(f64, f64)>{
        let training_time = Instant::now();

        let vs = nn::VarStore::new(self.device.get());
        self.model.replace(network(&vs.root(),
                                   self.config.image_dim, 
                                   self.config.labels, 
                                   self.config.units,
                                   hidden_layers.clone()));
        let mut optimizer = nn::Adam::default()
            .build(&vs, self.config.learning_rate)?;
        let mut loss = Tensor::zeros([0],(Kind::Float, vs.device()));

        let dataset: Vec<_> = if self.config.shuffle{
            self.dataset.train_iter(self.config.batch_size)
                .shuffle().to_device(vs.device()).collect()
        }
        else{
            self.dataset.train_iter(self.config.batch_size)
                .to_device(vs.device()).collect()
        };

        for _ in 0..self.config.epochs {  
            for (images, labels) in &dataset{
                loss = self.model.borrow_mut().forward_t(&images,true)
                    .cross_entropy_for_logits(&labels);            
                optimizer.zero_grad();
                optimizer.backward_step(&loss);
            }
        }
        Ok((loss.double_value(&[]), training_time.elapsed().as_secs_f64()))
    }

    fn test_model(&self) -> Result<(f64, f64, f64)>{ 
        let test_time = Instant::now();
        let mut loss = 0.;
        let mut accuracy = 0.;
        let mut size = 0;

        let dataset :Vec<_> = if self.config.shuffle{
            self.dataset.test_iter(self.config.test_batch_size)
                .shuffle().to_device(self.device.get()).collect()
        }
        else{
            self.dataset.test_iter(self.config.test_batch_size)
                .to_device(self.device.get()).collect()
        };

        for (images, labels) in &dataset{
            let out_validation = self.model.borrow_mut().forward_t(&images, false);
            loss += out_validation.cross_entropy_for_logits(&labels).double_value(&[]);
            accuracy += out_validation.accuracy_for_logits(&labels).double_value(&[]);
            size += 1;
        }
        let size = size as f64;

        Ok((100.*accuracy/size, loss/size, test_time.elapsed().as_secs_f64()))
    }

    fn run_on_device(&self, cuda: bool) -> Result<()>{
        let device_name;
        let layers_list;
        if !cuda {
            device_name = "cpu";
            self.device.set(Device::Cpu);
            layers_list = &self.config.layers_cpu;
        }
        else if cuda{
            device_name = "cuda";
            self.device.set(Device::Cuda(0));
            layers_list = &self.config.layers_cuda;
        }
        else{
            panic!("Unknown device");        
        };

        let labels = vec!["hidden_layers", "run", "training_loss", 
        "training_time", "test_loss", "test_accuracy", 
        "test_time"];
        let mut dir_path: String = self.config.log_dir.to_owned();

        dir_path.push_str("/rust_log_");
        dir_path.push_str(&device_name);
        dir_path.push_str(".csv");

        let mut log_csv = LogCSV::new(dir_path, labels.clone());
        let log_console = LogConsole::new(labels.clone());


        for layers in layers_list{
            for test in 0..self.config.runs{
                let Ok((loss, training_time))  = self.train_epoch(layers) 
                    else { panic!("Unknown error")};
                let Ok((acc, test_loss, test_time)) = self.test_model() 
                    else {panic!("Unknown error")};

                let data = vec![layers.to_string(), 
                                test.to_string(),
                                loss.to_string(), 
                                training_time.to_string(),
                                test_loss.to_string(), 
                                acc.to_string(),
                                test_time.to_string()]; 

                log_console.log(data.clone());
                log_csv.log(data.clone());
            }
        }
        Ok(())
    }

    pub fn run(&self){
        for cuda in &self.config.use_cuda{
            let _  = self.run_on_device(cuda.clone());
        }
    }

    pub fn new(configuration: Settings) -> LearningAlgorithm{
        let mut dir_path: String = configuration.data_path.to_owned();
        dir_path.push_str("/MNIST/raw");

        let Ok(data) = tch::vision::mnist::load_dir(dir_path)
            else { panic!("Data not found")};
            LearningAlgorithm{
                config: configuration,
                dataset: data,
                device: Cell::new(Device::Cpu),
                model: RefCell::new(nn::seq()),
            }
    }
}
