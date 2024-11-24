import { DrawingPad } from "../components/drawingPad";
import { PredictionResult } from "../components/predictionResult";
import { useState, useEffect } from "react";


const API_BASE = "http://127.0.0.1:5000"

export const HomePage = () => {
    const [prediction, setPrediction] = useState("")

    const postImage = async (imageInfo) => {
        const data = await fetch(API_BASE + "/predict", {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({
              image: imageInfo
            }),
          }).then((res) => res.json())
            .then(data => setPrediction(data.prediction))
            .catch(error => console.error('Error: ', error))
    }
  
    return (
        <div className="">
            <DrawingPad postImage={postImage}/>
            <PredictionResult />
        </div>
    );
}