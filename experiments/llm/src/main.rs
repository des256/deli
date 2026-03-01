use {
    base::*,
    inference::*,
    std::io::Write,
};

use inference::{gemma3, llama3, phi3};

enum Model {
    Phi3(phi3::Phi3Handle<()>, phi3::Phi3Listener<()>),
    Llama3B(llama3::Llama3Handle<()>, llama3::Llama3Listener<()>),
    Llama8B(llama3::Llama3Handle<()>, llama3::Llama3Listener<()>),
    Gemma4B(gemma3::Gemma3Handle<()>, gemma3::Gemma3Listener<()>),
    Gemma12B(gemma3::Gemma3Handle<()>, gemma3::Gemma3Listener<()>),
}

fn format_prompt(model: &Model, user_text: &str) -> String {
    match model {
        Model::Phi3(..) => format!(
            "<|system|>\nYou are a helpful assistant.<|end|>\n<|user|>\n{}<|end|>\n<|assistant|>\n",
            user_text,
        ),
        Model::Llama3B(..) | Model::Llama8B(..) => format!(
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            user_text,
        ),
        Model::Gemma4B(..) | Model::Gemma12B(..) => format!(
            "<start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\n",
            user_text,
        ),
    }
}

fn send_prompt(model: &Model, prompt: String) -> Result<(), InferError> {
    let input = LlmInput { payload: (), prompt };
    match model {
        Model::Phi3(h, _) => h.send(input).map_err(|_| InferError::Runtime("send failed".into())),
        Model::Llama3B(h, _) | Model::Llama8B(h, _) => {
            h.send(input).map_err(|_| InferError::Runtime("send failed".into()))
        }
        Model::Gemma4B(h, _) | Model::Gemma12B(h, _) => {
            h.send(input).map_err(|_| InferError::Runtime("send failed".into()))
        }
    }
}

async fn recv_token(model: &mut Model) -> Option<Stamped<LlmOutput<()>>> {
    match model {
        Model::Phi3(_, l) => l.recv().await,
        Model::Llama3B(_, l) | Model::Llama8B(_, l) => l.recv().await,
        Model::Gemma4B(_, l) | Model::Gemma12B(_, l) => l.recv().await,
    }
}

#[tokio::main]
async fn main() -> Result<(), InferError> {
    base::init_stdout_logger();

    println!("Select an LLM:");
    println!("  1. Phi-3");
    println!("  2. Llama 3 (3B)");
    println!("  3. Llama 3 (8B)");
    println!("  4. Gemma 3 (4B)");
    println!("  5. Gemma 3 (12B)");
    print!("> ");
    std::io::stdout().flush().map_err(|e| InferError::Runtime(e.to_string()))?;

    let mut choice = String::new();
    std::io::stdin()
        .read_line(&mut choice)
        .map_err(|e| InferError::Runtime(e.to_string()))?;

    let choice = choice.trim();
    let model_name = match choice {
        "1" => "Phi-3",
        "2" => "Llama 3 (3B)",
        "3" => "Llama 3 (8B)",
        "4" => "Gemma 3 (4B)",
        "5" => "Gemma 3 (12B)",
        _ => {
            println!("Invalid choice: {}", choice);
            return Ok(());
        }
    };

    println!("Loading {}...", model_name);
    let inference = Inference::new()?;
    let epoch = Epoch::new();
    let executor = onnx::Executor::Cuda(0);

    let mut model = match choice {
        "1" => {
            let (h, l) = inference.use_phi3(&executor, epoch)?;
            Model::Phi3(h, l)
        }
        "2" => {
            let (h, l) = inference.use_llama3_3b(&executor, epoch)?;
            Model::Llama3B(h, l)
        }
        "3" => {
            let (h, l) = inference.use_llama3_8b(&executor, epoch)?;
            Model::Llama8B(h, l)
        }
        "4" => {
            let (h, l) = inference.use_gemma3_4b(&executor, epoch)?;
            Model::Gemma4B(h, l)
        }
        "5" => {
            let (h, l) = inference.use_gemma3_12b(&executor, epoch)?;
            Model::Gemma12B(h, l)
        }
        _ => unreachable!(),
    };

    println!("{} loaded. {}", model_name, inference.mem_info());
    println!("Type a message and press Enter. Ctrl+D to exit.\n");

    loop {
        print!("> ");
        std::io::stdout().flush().map_err(|e| InferError::Runtime(e.to_string()))?;

        let mut input = String::new();
        std::io::stdin()
            .read_line(&mut input)
            .map_err(|e| InferError::Runtime(e.to_string()))?;
        if input.is_empty() {
            break;
        }

        let input = input.trim();
        if input.is_empty() {
            continue;
        }

        let prompt = format_prompt(&model, input);
        send_prompt(&model, prompt)?;

        while let Some(stamped) = recv_token(&mut model).await {
            match stamped.inner {
                LlmOutput::Token { token, .. } => {
                    print!("{}", token);
                    std::io::stdout().flush().ok();
                }
                LlmOutput::Eos { .. } => {
                    println!();
                    break;
                }
            }
        }
    }

    Ok(())
}
