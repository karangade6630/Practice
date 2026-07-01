import dotenv from "dotenv";
dotenv.config();
import express from "express";
import {GoogleGenAI} from "@google/genai";


// Initialize the GoogleGenAI client with your API key
const ai  = new GoogleGenAI({
    apiKey: process.env.GEMINI_API_KEY,
})
// Initialize the Express app
const app = express();


// Define a route to handle the streaming response from the Gemini API
app.get("/",(res,req)=>{
    req.status(200).json({message:"ok",status:200})
})
app.get("/stream",async(req,res)=>{
try {

        const responseStream = await ai.models.generateContentStream({
            model:"gemini-2.5-flash",
            contents:"What is programming?",
            config:{
                systemInstruction:"You are a helpful assistant that provides concise and clear explanations in detailed 1000 words.",
            }
        })
        for await (const chunk of responseStream){
            if(chunk){
              res.write(chunk.text);
            }
        }
        res.end();
} catch (error) {
    console.log("Error in getResponse function: ", error);
    res.status(500).send("Error in getResponse function: " + error.message);
}
})



app.listen(3000, () => {
    console.log("Server is running on http://localhost:3000");
})