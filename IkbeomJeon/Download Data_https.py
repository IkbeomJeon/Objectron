import requests
import os

class_names = [
    "book",
    "shoe",
    "chair",
    "cup",
    "bottle",


]

default_download_directory = "../../Datasets/"

for class_name in class_names:
    public_url = "https://storage.googleapis.com/objectron"
    blob_path = f"{public_url}/v1/index/{class_name}_annotations"
    video_ids = requests.get(blob_path).text
    video_ids = video_ids.split('\n')

    # Download the first ten videos in cup test dataset

    for i in range(0, len(video_ids)):

        id = video_ids[i]

        video_filename = public_url + "/videos/" + id + "/video.MOV"
        metadata_filename = public_url + "/videos/" + id + "/geometry.pbdata"
        annotation_filename = public_url + "/annotations/" + id + ".pbdata"

        # video.content contains the video file.
        video = requests.get(video_filename)
        metadata = requests.get(metadata_filename)
        annotation = requests.get(annotation_filename)

        download_directory_video = f"{default_download_directory}/videos/{id}"
        os.makedirs(download_directory_video, exist_ok=True)


        #video
        video_file_path = f"{download_directory_video}/video.MOV"
        file = open(video_file_path, "wb")
        file.write(video.content)
        file.close()

        #meta
        metadata_file_path = f"{download_directory_video}/geometry.pbdata"
        file = open(metadata_file_path, "wb")
        file.write(metadata.content)
        file.close()


        #annotation
        download_directory_annotation = f"{default_download_directory}/annotations/{id}"
        os.makedirs(os.path.abspath(download_directory_annotation+'/../'), exist_ok=True)
        annotation_file_path = f"{download_directory_annotation}.pbdata"
        file = open(annotation_file_path, "wb")
        file.write(annotation.content)
        file.close()

        print("download done %d/%d (%s) : " % (i + 1, len(video_ids), id))
        if i == 10 :
            break

