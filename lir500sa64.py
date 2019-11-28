import ctypes
import numpy as np

class LIR500SA:
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height

        self.lir500sa = ctypes.cdll.LoadLibrary('LIR500SA_SDK_ST64.dll')

        self.LIR500SA_ST_FrameBuffer = ctypes.c_ubyte * width * height
        self.frame_buffer = self.LIR500SA_ST_FrameBuffer()

        self.LIR500SA_ST_Open = self.lir500sa['LIR500SA_ST_Open']
        self.LIR500SA_ST_Open.argtypes = None
        self.LIR500SA_ST_Open.restype = ctypes.c_void_p

        self.LIR500SA_ST_Close = self.lir500sa['LIR500SA_ST_Close']
        self.LIR500SA_ST_Close.argtypes = (ctypes.c_void_p,)
        self.LIR500SA_ST_Close.restype = None

        self.LIR500SA_ST_Connect = self.lir500sa['LIR500SA_ST_Connect']
        self.LIR500SA_ST_Connect.argtypes = (ctypes.c_void_p, ctypes.c_char_p,);
        self.LIR500SA_ST_Connect.restype = ctypes.c_bool

        self.LIR500SA_ST_Disconnect = self.lir500sa['LIR500SA_ST_Disconnect']
        self.LIR500SA_ST_Disconnect.argtypes = (ctypes.c_void_p,);
        self.LIR500SA_ST_Disconnect.restype = ctypes.c_bool

        self.LIR500SA_ST_SetRange = self.lir500sa['LIR500SA_ST_SetRange']
        self.LIR500SA_ST_SetRange.argtypes = (ctypes.c_void_p, ctypes.c_float, ctypes.c_float,);
        self.LIR500SA_ST_SetRange.restype = ctypes.c_bool

        self.LIR500SA_ST_GetFrame = self.lir500sa['LIR500SA_ST_GetFrame']
        self.LIR500SA_ST_GetFrame.argtypes = (ctypes.c_void_p, self.LIR500SA_ST_FrameBuffer, ctypes.c_int,);
        self.LIR500SA_ST_GetFrame.restype = ctypes.c_int

        self.handle = self.LIR500SA_ST_Open()

    def __del__(self):
        if hasattr(self, 'LIR500SA_ST_Close'):
            self.LIR500SA_ST_Close(self.handle)

    def connect(self, ip):
        return self.LIR500SA_ST_Connect(self.handle, ip)

    def disconnect(self):
        return self.LIR500SA_ST_Disconnect(self.handle)

    def set_range(self, min, max):
        return self.LIR500SA_ST_SetRange(self.handle, min, max)

    def get_frame(self):
        ret = self.LIR500SA_ST_GetFrame(self.handle, self.frame_buffer, self.width*self.height)
        if ret > 0:
            img = np.frombuffer(self.frame_buffer, np.uint8).reshape(self.height, self.width, 1)
            # 스택으로 수정
            # stacked_img = np.stack((img,)*3, axis=-1)
            stacked_img = np.squeeze(np.stack((img,) * 3, -1))

            return stacked_img
        elif ret == 0:    # empty frame
            return np.array([])
        
        return None;    # disconnected
