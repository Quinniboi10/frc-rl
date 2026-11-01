use reqwest::blocking::Client;
use serde_json::{Value, json};
use rand::random_range;
use std::{thread, time};

const MAX_MS: u64 = 100; // Test more req/s until this avg time

const REQUESTS_PER_THREAD: u64 = 100;

const PORT: i32 = 5005;
const NUM_RAYS: i32 = 65;

fn random() -> f32 {
    return random_range(10..100) as f32 / 10.0;
}

fn post(client: &Client, endpoint: &str, body: &Value) {
    client.post(format!("http://127.0.0.1:{}{}", PORT, endpoint))
    .header("Content-Type", "Application/json").json(&body).send().ok().unwrap();
}

fn reset(client: &Client, id: i32) {
    let endpoint = format!("/reset/{}/", id);
    let x = random();
    let y = random();
    let body = json!({"x": x, "y": y});

    post(client, &endpoint, &body);
}

fn update(client: &Client, id: i32) -> u128 {
    let endpoint = format!("/update/{}/", id);

    let distances: Vec<f32> = (0..NUM_RAYS).map(|_| random()).collect();
    
    let body = json!({
        "vel_fwd": random(),
        "vel_right": random(),

        "pos_x": random(),
        "pos_y": random(),

        "vel_angular": random(),
        "rot_sin": random(),
        "rot_cos": random(),

        "raycast_distances": distances
    });

    let start = time::Instant::now();
    post(client, &endpoint, &body);
    return (time::Instant::now() - start).as_millis();
}

fn simulate_client(id: i32) -> f64 {
    let client = Client::new();

    reset(&client, id);

    let mut next_frame = time::Instant::now();
    let frame_time = time::Duration::from_millis(100); // 10hz

    let mut total_ms = 0u128;

    for _ in 0..REQUESTS_PER_THREAD {
        total_ms += update(&client, id);
        next_frame += frame_time;
        thread::sleep(next_frame.saturating_duration_since(time::Instant::now()));
    }

    return total_ms as f64 / REQUESTS_PER_THREAD as f64;
}

fn main() {
    let mut req_per_s = 10;

    loop {
        let n_threads = req_per_s / 10;

        let handles: Vec<_> = (0..n_threads).map(|id| thread::spawn(move || simulate_client(id as i32))).collect();

        let avg_time = handles.into_iter().map(|handle| handle.join().unwrap()).sum::<f64>() / n_threads as f64;
        println!("Average response time for {} r/s: {:2}ms", req_per_s, avg_time);

        if avg_time > MAX_MS as f64 {
            break;
        }

        req_per_s += 10;
    }
}
