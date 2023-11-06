use anyhow::Result;
use tch::{nn,Device, nn::Module, nn::OptimizerConfig,Tensor,Kind, no_grad};
use std::{time::Instant, cell::Cell, iter::Iterator}; 

use crate::utils::network::network;
use crate::settings::Settings;
use crate::utils::logger::{LogConsole, LogCSV};

pub struct LearningAlgorithm{
    config: Settings,
    train_dataset: Vec<(Tensor,Tensor)>,
    test_dataset: Vec<(Tensor,Tensor)>,
    device: Cell<Device>,
}

impl LearningAlgorithm{    
    fn train_epoch(&mut self, hidden_layers: &i64) -> Result<(impl Module, f64, f64)>{
        let vs = nn::VarStore::new(self.device.get());
        let net = network(&vs.root(), self.config.image_dim, 
                          self.config.labels, 
                          self.config.units,
                          hidden_layers.clone());
        let mut optimizer = nn::Adam::default()
                                     .build(&vs, self.config.learning_rate)?;
        let mut loss = Tensor::zeros([0],(Kind::Float, self.device.get()));
        let training_time = Instant::now();
        
        for _ in 0..self.config.epochs {  
            for (images, labels) in self.train_dataset.iter(){

                let images = &images.to(self.device.get());
                let labels = &labels.to(self.device.get());
                loss = net.forward(&images)
                          .cross_entropy_for_logits(&labels);            
                optimizer.backward_step(&loss);
            }
        }

        Ok((net ,loss.double_value(&[]), training_time.elapsed().as_secs_f64()))
    }

    fn test_model(&mut self, net: impl Module) -> Result<(f64, f64, f64)>{ 
        // let size = self.dataset.test_images.size()[0] as f64;
        let mut loss = 0.;
        let mut accuracy = 0.;
        let test_time = Instant::now();

        let mut size : f64 = 0.;
        for (images, labels) in self.test_dataset.iter(){

            let images = &images.to(self.device.get());
            let labels = &labels.to(self.device.get());
            no_grad(|| {
                let out_validation = net.forward(&images);
                loss += out_validation.cross_entropy_for_logits(&labels).double_value(&[]);
                accuracy += 100.*out_validation.accuracy_for_logits(&labels).double_value(&[]);
            });
            size += 1.;
        }

        Ok((accuracy/size, loss/size, test_time.elapsed().as_secs_f64()))
    }

    fn run_on_device(&mut self, cuda: bool) -> Result<()>{
        let device_name;
        let layers_list;
        if !cuda {
            device_name = "cpu";
            self.device.set(Device::Cpu);
            layers_list = self.config.layers_cpu.clone();
        }
        else if cuda{
            device_name = "cuda";
            self.device.set(Device::Cuda(0));
            layers_list = self.config.layers_cuda.clone();
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


        for layers in &layers_list{
            for test in 0..self.config.runs{
                let Ok((net, loss, training_time))  = self.train_epoch(layers) 
                    else { panic!("Unknown error")};
                let Ok((acc, test_loss, test_time)) = self.test_model(net) 
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

    pub fn run(&mut self){
        for cuda in self.config.use_cuda.clone(){
            let _  = self.run_on_device(cuda.clone());
        }
    }

    pub fn new(configuration: Settings) -> LearningAlgorithm{
        let mut dir_path: String = configuration.data_path.to_owned();
        dir_path.push_str("/MNIST/raw");

        let Ok(data) = tch::vision::mnist::load_dir(dir_path) 
            else { panic!("Data not found")};
        
        let(train_dataset, test_dataset) = 
        if configuration.shuffle{
            (data.train_iter(configuration.batch_size).shuffle().collect::<Vec<_>>(),
            data.test_iter(configuration.test_batch_size).shuffle().collect::<Vec<_>>())
        }
        else{
            (data.train_iter(configuration.batch_size).collect::<Vec<_>>(),
            data.test_iter(configuration.test_batch_size).collect::<Vec<_>>())
        };

        LearningAlgorithm{
            config: configuration,
            train_dataset,
            test_dataset,
            device: Cell::new(Device::Cpu),
        }
    }
}
