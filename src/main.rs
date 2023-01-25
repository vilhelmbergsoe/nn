use ndarray::{arr1, Array1, Array2};
use rand::distributions::{Distribution, Uniform};
use fastapprox::fast::tanh;

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
            w: Array2::from_shape_simple_fn((p, c), rand::random),
            b: Array1::from_shape_simple_fn(c, rand::random),
        }
    }
    pub fn calc(&self, inputs: &Array1<f32>) -> Array1<f32> {
        (inputs.dot(&self.w) + &self.b).map(|x| tanh(*x))
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

        for layer in &self.layers {
            curr = layer.calc(&curr);
        }

        curr
    }
    pub fn mutate(&mut self, mut_rate: f32, mut_w_size: f32, mut_b_size: f32) {
        let mut rng = rand::thread_rng();
        for i in 0..self.layers.len() {
            self.layers[i].w = self.layers[i].w.map(|x| {
                if rand::random::<f32>() <= mut_rate {
                    let mut_size = Uniform::from(0.0 - (mut_w_size / 2.0)..=0.0 + (mut_w_size / 2.0));
                    let v = mut_size.sample(&mut rng);
                    x + v
                } else {
                    *x
                }
            });
            self.layers[i].b = self.layers[i].b.map(|x| {
                if rand::random::<f32>() <= mut_rate {
                    let mut_size = Uniform::from(0.0 - (mut_b_size / 2.0)..=0.0 + (mut_b_size / 2.0));
                    let v = mut_size.sample(&mut rng);
                    x + v
                } else {
                    *x
                }
            });
        }
    }
    pub fn print(&self) {
        for layer in &self.layers {
            print!("weights:\t");
            layer.w.for_each(|x| {
                print!("{x}\t");
            });
            println!();
            print!("biases:\t");
            layer.b.for_each(|x| {
                print!("{x}\t");
            });
            println!();
        }
    }
    pub fn fitness(&self, inputs: &[Array1<f32>], outputs: &[f32]) -> f32 {
        let mut diffs: Vec<f32> = Vec::with_capacity(inputs.len());

        for i in 0..inputs.len() {
            let diff = self.predict(&inputs[i])[0] - outputs[i];

            diffs.push(diff*diff);
        }

        let sum = diffs.iter().sum::<f32>();

        sum / diffs.len() as f32
    }
}

fn get_best(pool: &mut [NeuralNetwork], inputs: &[Array1<f32>], outputs: &[f32]) {
    radsort::sort_by_cached_key(pool, |nn| nn.fitness(inputs, outputs));
}

fn test_fn(x: f32) -> f32 {
    // x.sin() / 2.0
    // x / 20.0
    sigmoid(x)
}

fn main() {
    let pool_size: usize = 500;

    let mut pool: Vec<NeuralNetwork> = Vec::new();

    for _ in 0..pool_size {
        pool.push(NeuralNetwork::new(&[1, 2, 3, 5, 5, 5, 3, 2, 1]));
    }

    let num_gens = 20000;

    for gen in 0..num_gens {
        let mut inputs: Vec<Array1<f32>> = Vec::with_capacity(20);
        let mut outputs: Vec<f32> = Vec::with_capacity(20);
        for i in 0..20 {
            // let mut rng = rand::thread_rng();
            // let r = Uniform::from(-10.0..=10.0);
            // let num = r.sample(&mut rng);
            let num = i as f32;

            inputs.push(arr1(&[num]));
            outputs.push(test_fn(num));
        }

        get_best(&mut pool, &inputs, &outputs);

        let top = 4;
        let len = pool.len()/top;
        for i in 0..len {
            for j in 0..top-1 {
                pool[i*(top-1)+j+len] = pool[i].clone();
            }
        }

        for i in (pool.len() / top)..pool.len() {
            pool[i].mutate(0.1, 0.5, 0.05);
        }

        println!("generation {gen}: {}", &pool[0].fitness(&inputs, &outputs));

        if gen == num_gens-1 {
            println!("---");

            pool[0].print();

            for i in 0..inputs.len() {
                println!(
                    "{}: {} ({})",
                    inputs[i],
                    pool[0].predict(&inputs[i]),
                    outputs[i]
                );
            }
        }
    }
}
