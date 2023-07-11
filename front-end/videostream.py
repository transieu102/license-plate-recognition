import cv2
import av
import asyncio
import base64
import websockets
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from av.video.frame import VideoFrame
class VideoStreamTrack(MediaStreamTrack):
    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path
        self.container = av.open(self.video_path)
        self.video_stream = self.container.streams.video[0]

    async def recv(self):
        for frame in self.container.decode(video=0):
            if isinstance(frame, VideoFrame):
                pts, time_base = frame.pts, frame.time_base
                frame_data = frame.to_ndarray(format='bgr24')
                ret, buffer = cv2.imencode('.jpg', frame_data)
                if ret:
                    image_data = buffer.tobytes()
                    pts, time_base = frame.pts, frame.time_base
                    duration = int(1000 * time_base * pts)
                    self.timestamp += duration
                    return VideoFrame(width=frame.width, height=frame.height, data=image_data, pts=pts,
                                      time_base=time_base)
        raise EOFError


async def offer_video(websocket, path):
    pc = RTCPeerConnection()

    video_track = VideoStreamTrack('test.mp4')  # Đường dẫn tới video của bạn
    pc.addTrack(video_track)

    await pc.setLocalDescription(await pc.createOffer())
    await websocket.send(pc.localDescription.sdp)

    remote_desc = await websocket.recv()
    await pc.setRemoteDescription(RTCSessionDescription(sdp=remote_desc, type='answer'))

    while True:
        frame = await video_track.recv()
        if frame:
            image_data = base64.b64encode(frame.data).decode('utf-8')
            await websocket.send(image_data)


async def start_server():
    server = websockets.serve(offer_video, 'localhost', 8765)
    await server


if __name__ == "__main__":
    asyncio.run(start_server())
