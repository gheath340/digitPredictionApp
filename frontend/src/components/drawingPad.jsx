import React, { useRef, useState, useEffect } from 'react';

export const DrawingPad = ({ postImage, clear }) => {

  const canvasRef = useRef(null);
  const [isDrawing, setIsDrawing] = useState(false);

  // Set the canvas size
  const canvasWidth = 280;
  const canvasHeight = 280;

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    // Set the canvas background to white
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  }, []);

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
    ctx.lineWidth = 5
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

  const clearCanvas = () => {
    clear(canvasRef)
  }

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
        <button className="border border-1 border-black transition-transform transform hover:scale-110 active:scale-95 rounded-lg focus:outline-none" onClick={clearCanvas}>Clear</button>
        <button className="border border-1 border-black transition-transform transform hover:scale-110 active:scale-95 rounded-lg focus:outline-none" onClick={download}>Capture</button>
    </div>
  );
};