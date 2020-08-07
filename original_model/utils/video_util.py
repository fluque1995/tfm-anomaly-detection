from utils.array_util import *
import parameters as params
import cv2


def get_video_clips(video_path):
    """Divides the input video into non-overlapping clips

    :param video_path: Path to the video
    :returns: Array with the fragments of video
    :rtype: np.ndarray

    """
    frames = get_video_frames(video_path)
    clips = sliding_window(frames, params.frame_count, params.frame_count)
    return clips, len(frames)


def get_video_frames(video_path):
    """Reads the video given a file path

    :param video_path: Path to the video
    :returns: Video as an array of frames
    :rtype: np.ndarray

    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            break
    cap.release()
    return frames
