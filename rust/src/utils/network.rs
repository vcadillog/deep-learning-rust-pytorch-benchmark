use tch::nn;

pub fn network(vs: &nn::Path, input_dim: i64, output_dim: i64, units: i64, hidden_layers: u32) -> nn::Sequential{
    let hidden_layers = hidden_layers as u32;
    if hidden_layers <= 0{ 
        panic!("Layers should be positive");
    }
    let mut seq = nn::seq();
    seq = seq.add(nn::linear(vs / "input_layer", 
            input_dim,
            units,
            Default::default(),
    ))
        .add_fn(|xs| xs.relu());

    if hidden_layers > 1 {
        let mut input_units;
        let mut output_units;

        for layers in 0..hidden_layers {            
            if layers < hidden_layers/2 {
                input_units = i64::pow(2,layers); 
                output_units = i64::pow(2,layers+1);
            } 
            else if hidden_layers%2 == 1 && layers == hidden_layers/2{
                input_units = i64::pow(2,layers); 
                output_units = input_units; 
            }
            else if layers >= hidden_layers/2{
                input_units = i64::pow(2,hidden_layers-layers); 
                output_units = i64::pow(2,hidden_layers-layers-1); 
            }
            else {
                panic!("Unknown error");
            }
            seq = seq.add(nn::linear(vs, 
                    input_units*units,
                    output_units*units, 
                    Default::default()))
                .add_fn(|xs| xs.relu());
            }
    }
    seq.add(nn::linear(vs / "output_layer", units, output_dim, Default::default()))
}

