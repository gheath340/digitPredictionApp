import React, { useRef, useState } from 'react';

export const DrawingPad = ({ postImage }) => {

  const canvasRef = useRef(null);
  const [isDrawing, setIsDrawing] = useState(false);

  // Set the canvas size
  const canvasWidth = 280;
  const canvasHeight = 280;

   // Get mouse position on canvas
    const getMousePos = (e) => {
    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    return { x, y };
  };

  // Start drawing
  const startDrawing = (e) => {
    e.preventDefault()
    const pos = e.touches ? getMousePos(e.touches[0]) : getMousePos(e);
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    ctx.beginPath();
    ctx.moveTo(pos.x, pos.y);
    setIsDrawing(true);
  };

  // Draw while moving the mouse
  const draw = (e) => {
    if (!isDrawing) return;
    e.preventDefault()
    const pos = e.touches ? getMousePos(e.touches[0]) : getMousePos(e);
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();
  };

  // Stop drawing
  const stopDrawing = () => {
    setIsDrawing(false);
  };

  // Clear the canvas
  const clearCanvas = () => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height); // Clear the canvas
  };
  // Captures the data inside the canvas, sends it to server to be predicted
  const download = () => {
    let canvas1 = canvasRef.current;
    let url = canvas1.toDataURL("image/png");
    postImage(url)
  }

  return (
    <div className="">
        <canvas
            ref={canvasRef}
            width={canvasWidth}
            height={canvasHeight}
            onMouseDown={startDrawing}
            onMouseMove={draw}
            onMouseUp={stopDrawing}
            onMouseLeave={stopDrawing}
            onTouchStart={startDrawing}
            onTouchMove={draw}
            onTouchEnd={stopDrawing}
            onTouchCancel={stopDrawing}
            style={{ 
                border: '1px solid black',
                touchAction: 'none',
            }}
        />
        <button onClick={clearCanvas}>Clear</button>
        <button onclick={download}>Capture</button>
    </div>
  );
};