import dotenv from "dotenv";
dotenv.config();
import {GoogleGenAI} from "@google/genai";

const ai = new GoogleGenAI({
    apiKey: process.env.GEMINI_API_KEY
})

// const getResponse = async ()=>{
// try{
//     const res = await ai.models.generateContent({
//         model:"gemini-2.5-flash",
//         contents:"What is programming?",
//         config:{
//             systemInstruction:"You are a helpful assistant that provides concise and clear explanations in short paragraphs in 100 words.",
//         }
//     })
//     console.log("Response from Gemini API: ", res.text);
// }catch(err){
// console.log("Error in getResponse function: ", err);
// }
// }



const getResponse = async ()=>{
try{
    const res = await ai.models.generateContentStream({
        model:"gemini-2.5-flash",
        contents:"What is programming?",
        config:{
            systemInstruction:"You are a helpful assistant that provides concise and clear explanations in short paragraphs in 500 words.",
        }
    })
    console.log("Response from Gemini API: ");
    for await (const chunk of res){
        if(chunk){
            console.log(chunk.text);
        }
    }

}catch(err){
console.log("Error in getResponse function: ", err);
}
}

getResponse();