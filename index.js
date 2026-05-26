import ollama from "ollama";

import fs from "fs";


const base64Image = fs.readFileSync("image.png", { encoding: "base64" });

async function main() {
  try {

    const stream = await ollama.chat({
      model: "qwen3.5:0.8b",

      messages: [
        {
          role: "user",
          content: "what is in this image?",
          images:[base64Image]
        },
      ],

      stream: true,

      options: {

        // Less repetition
        repeat_penalty: 1.5,
        repeat_last_n: 256,

        // Better creativity
        temperature: 0.7,
        top_p: 0.9,
        top_k: 40,

        // More stable output
        num_predict: 200,

        // Stop rambling
        stop: [
          "<|im_end|>",
          "<|endoftext|>"
        ]
      },

      // Disable reasoning/thinking
      think: false,
    });

    let started = false;

    for await (const chunk of stream) {

      if (chunk.message.content) {

        if (!started) {
          console.log("\n💬 Response:\n");
          started = true;
        }

        process.stdout.write(chunk.message.content);
      }
    }

    console.log("\n");

  } catch (error) {
    console.error("Error:", error);
  }
}

main();