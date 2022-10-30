use ndarray::{arr1, Array1, Array2};
use rand::distributions::{Distribution, Uniform};

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn rsigmoid(y: f32) -> f32 {
    -((1.0 / y) - 1.0).ln()
}

#[derive(Clone)]
struct NeuralLayer {
    w: Array2<f32>,
    b: Array1<f32>,
}

impl NeuralLayer {
    pub fn new_rand(p: usize, c: usize) -> Self {
        NeuralLayer {
            w: Array2::from_shape_simple_fn((p, c), || rand::random()),
            b: Array1::from_shape_simple_fn(c, || rand::random()),
        }
    }
    pub fn calc(&self, inputs: Array1<f32>) -> Array1<f32> {
        (inputs.dot(&self.w) + &self.b).map(|x| x.tanh())
    }
}

#[derive(Clone)]
struct NeuralNetwork {
    layers: Vec<NeuralLayer>,
}

impl NeuralNetwork {
    pub fn new(layer_sizes: &[usize]) -> Self {
        let mut layers: Vec<NeuralLayer> = Vec::new();

        for i in 1..layer_sizes.len() {
            layers.push(NeuralLayer::new_rand(layer_sizes[i - 1], layer_sizes[i]));
        }

        NeuralNetwork { layers }
    }
    pub fn predict(&self, inputs: &Array1<f32>) -> Array1<f32> {
        let mut curr: Array1<f32> = inputs.clone();

        for layer in self.layers.iter() {
            curr = layer.calc(curr);
        }

        curr
    }
    pub fn mutate(&mut self, mut_rate: f32, mut_size: f32) {
        let mut rng = rand::thread_rng();
        for i in 0..self.layers.len() {
            self.layers[i].w = self.layers[i].w.map(|x| {
                if rand::random::<f32>() <= mut_rate {
                    let mut_size = Uniform::from(0.0 - (mut_size / 2.0)..=0.0 + (mut_size / 2.0));
                    let v = mut_size.sample(&mut rng);
                    x + v
                } else {
                    *x
                }
            });
            self.layers[i].b = self.layers[i].b.map(|x| {
                if rand::random::<f32>() <= mut_rate {
                    let mut_size = Uniform::from(0.0 - (mut_size / 2.0)..=0.0 + (mut_size / 2.0));
                    let v = mut_size.sample(&mut rng);
                    x + v
                } else {
                    *x
                }
            });
        }
    }
    pub fn print(&self) {
        for layer in self.layers.iter() {
            print!("weights:\t");
            layer.w.for_each(|x| {
                print!("{}\t", x);
            });
            println!("");
            print!("biases:\t");
            layer.b.for_each(|x| {
                print!("{}\t", x);
            });
            println!("");
        }
    }
}

fn fitness(nn: &NeuralNetwork, inputs: &[Array1<f32>], outputs: &[f32]) -> f32 {
    let mut diffs: Vec<f32> = Vec::with_capacity(inputs.len());

    for i in 0..inputs.len() {
        let diff = nn.predict(&inputs[i])[0] - outputs[i];

        diffs.push(diff * diff);
    }

    let sum = diffs.iter().sum::<f32>();

    sum / diffs.len() as f32
}

fn get_best(pool: Vec<NeuralNetwork>, inputs: &[Array1<f32>], outputs: &[f32]) -> NeuralNetwork {
    let mut curr_best: NeuralNetwork = pool[0].clone();
    let mut curr_val: f32 = 100.0;

    for nn in pool {
        if fitness(&nn, inputs, outputs) < curr_val {
            curr_best = nn;
            curr_val = fitness(&curr_best, inputs, outputs);
        }
    }

    curr_best
}

fn test_fn(x: f32) -> f32 {
    x.sin() / 2.0
}

fn main() {
    let pool_size: usize = 500;

    let mut pool: Vec<NeuralNetwork> = Vec::new();

    for _ in 0..pool_size {
        pool.push(NeuralNetwork::new(&[1, 5, 3, 1]))
    }

    let num_gens = 10000;

    for gen in 0..num_gens {
        let mut inputs: Vec<Array1<f32>> = Vec::with_capacity(20);
        let mut outputs: Vec<f32> = Vec::with_capacity(20);
        for _ in 0..20 {
            let mut rng = rand::thread_rng();
            let r = Uniform::from(-10.0..=10.0);
            let num = r.sample(&mut rng);

            inputs.push(arr1(&[num]));
            outputs.push(test_fn(num));
        }

        let best = get_best(pool.clone(), &inputs, &outputs);

        pool[0] = best.clone();
        for i in 1..pool_size {
            pool[i] = best.clone();
            pool[i].mutate(0.1, 0.1);
        }

        println!("generation {}: {}", gen, fitness(&best, &inputs, &outputs));

        if gen == num_gens-1 {
            println!("---");

            best.print();

            for i in 0..inputs.len() {
                println!(
                    "{}: {} - {}",
                    inputs[i],
                    best.predict(&inputs[i]),
                    outputs[i]
                );
            }
        }
    }
}
