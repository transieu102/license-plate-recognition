$=document.querySelector.bind(document)
$$=document.querySelectorAll.bind(document)
// const SimplePeer = require('simple-peer');
const frameContainer=$('.frameContainer')
const inputSrc=$('.inputSrc')
const openInput=()=>{
    frameContainer.style.display='none'
    inputSrc.style.display='flex'
}
const backIcon=$('.backIcon')
backIcon.onclick=()=>{
    socket1.close()
    socket2.close()
    openInput()}
    const openFrame=()=>{
    frameContainer.style.display='flex'
    inputSrc.style.display='none'
}
const canvasOn=()=>{
    var canvas=$$('canvas')
    canvas.forEach((e)=>e.style.opacity=1)
}
const canvasOff=()=>{
    setTimeout(function() {
        var canvas=$$('canvas')
    canvas.forEach((e)=>e.style.opacity=.7)
    }, 2000);
    
}
const resetCanvas=()=>{
    const canvass = $$('canvas');
    canvass.forEach((can)=>{
    can.getContext('2d').clearRect(0, 0, can.width, can.height);})
}
var socket1 = new WebSocket('ws://192.168.20.156:8083/ws');
const VideoProcess=()=>{
        // Establish WebSocket connection
        socket1 = new WebSocket('ws://192.168.20.156:8083/ws');
    // When the connection is established
        socket1.onopen = function(event) {
            console.log('WebSocket connection established');
        };

// When a message is received from the backend
socket1.onmessage = function(event) {
    const frameData = event.data;
    // console.log(frameData)
    const parsedData = JSON.parse(frameData);
    console.log(parsedData['text'])
    displayFrame(parsedData['frame']);
};
}

// Function to display a frame on the canvas
function displayFrame(frameData) {
    const canvas = document.getElementById('canvasElement2');
    const context = canvas.getContext('2d');

    // Create an image element and set its source to the received frame data
    const img = new Image();
    img.src = 'data:image/jpeg;base64,'+frameData;

    // When the image is loaded, draw it on the canvas
    img.onload = function() {
        canvas.width = img.width;
        canvas.height = img.height;
        context.drawImage(img, 0, 0);
    };
}

// When a video file is selected
document.getElementById('videoInput').addEventListener('change', function(event) {
    const file = event.target.files[0];
    if (file && file.type.includes('video')){
        VideoProcess()
        // socket2.close()
        console.log('Upload Video successful')
        // processSocket1()
        resetCanvas()
        openFrame()
        // Create a URL for the selected video file
    const videoURL = URL.createObjectURL(file);

    // Set the video element source to the created URL
    const video = document.createElement('video');
    video.src = videoURL;

    // When the video is loaded
    video.onloadedmetadata = function() {
        const canvas = document.getElementById('canvasElement');
        const context = canvas.getContext('2d');
        canvasOn()
        // Set the canvas dimensions to match the video dimensions
        canvas.width = 500;
        canvas.height = 500;

        // Play the video
        video.play();

        // Process and send frames to the backend at a specified interval
        const frameInterval = 1000 / 30; // 30 frames per second
        const intervalId = setInterval(function() {
            if (video.paused || video.ended) {
                // canvasOff()
                clearInterval(intervalId)
                document.getElementById('videoInput').value=''
              }
            // Draw the current frame on the canvas
            context.drawImage(video, 0, 0, canvas.width, canvas.height);

            // Get the frame data from the canvas
            const frameData = canvas.toDataURL('image/jpeg', 0.8);

            // Send the frame data to the backend via WebSocket
            socket1.send(frameData);
        }, frameInterval);
        backIcon.onclick=()=>{
            openInput()
            clearInterval(intervalId)
            document.getElementById('videoInput').value=''}
    };
    }
    
});

const inputVideo=$('#inputVideo')
inputVideo.onclick=()=>{
    inputVideo.querySelector('input').click();
}
const inputStream=$('#inputStreamLink')
const popUp=$('.popUp')
inputStream.onclick=()=>{
    popUp.style.display='flex'
}
popUp.querySelector('.closeIcon').onclick=()=>{
    popUp.style.display="none"
}
popUp.querySelector('button').onclick=()=>{
    let url=document.getElementById("urlInput").value
    popUp.style.display="none"
    console.log(url)
    startStream(url)
}

var socket2 = new WebSocket('ws://192.168.20.156:8083/ws_stream');
const Stream=(url)=>{
    // Establish WebSocket connection
    socket2 = new WebSocket('ws://192.168.20.156:8083/ws_stream');
//  socket1.close()
// console.log(`Start connect to ${url}`)
// When the connection is established
socket2.onopen = function(event) {
    console.log('WebSocket connection established');
    socket2.send(url);
 };
 
 // When a message is received from the backend
 socket2.onmessage = function(event) {
     const frameData = event.data;
     // console.log(frameData)
     const parsedData = JSON.parse(frameData);
     console.log(parsedData['text'])
     displayFrame(parsedData['frame']);
 };
} 

const startStream=(url)=>{
    resetCanvas()
    openFrame()
    canvasOn()
    Stream(url)
}
