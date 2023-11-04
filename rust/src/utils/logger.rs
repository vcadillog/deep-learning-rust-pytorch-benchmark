use std::iter::zip;
use std::cmp::min;
use std::fs::File;
use csv;

pub struct LogCSV{
    file: csv::Writer<File>,
}

impl LogCSV{
    pub fn new (path: String , labels: Vec<&str>) -> LogCSV {
        let Ok(mut wtr) = csv::Writer::from_path(path) else {panic!("Path not found")};
        let _ = wtr.write_record(labels);
        LogCSV{
            file: wtr,
        } 
    } 
    pub fn log(&mut self, data: Vec<String>){
        let _ = self.file.write_record(data);
        let _ = self.file.flush();
    }
}

pub struct LogConsole<'a>{
    labels: Vec<&'a str>, 
}

impl <'a> LogConsole<'a>{
    pub fn new(labels: Vec<&str>) -> LogConsole{
        LogConsole{
            labels,
        }
    }
    pub fn log(&self, data: Vec<String>){
        let mut size = if self.labels.len() != data.len()
        {
            println!("The number of labels doesn't match with the data");
            min(self.labels.len(), data.len())
        }
        else{
            self.labels.len()
        };
        let mut buffer : String = "".to_owned();
        for (name, values) in zip(&self.labels, data){
            size -= 1;
            buffer.push_str(name);
            buffer.push_str(": ");
            buffer.push_str(values.as_str());
            if size > 0{
                buffer.push_str(" || ");
            }
        }

        println!("{:?}", buffer);
    }


}

