import youtube_transcript_api._errors

from tools.llama_index import wikipedia_search
from pydantic import BaseModel

class Foo(BaseModel):
    a: str
    b: int

def test_wikipedia_search():
    result = wikipedia_search('wikipedia')
    print(result)
    assert(type(result) is str)

def test_yt_trsc():
    from llama_index.readers.youtube_transcript import YoutubeTranscriptReader
    from llama_index.readers.youtube_transcript.utils import is_youtube_video

    loader = YoutubeTranscriptReader()
    url ="https://www.youtube.com/watch?v=L1vXCYZAYYM"
    if not is_youtube_video(url):
        raise ValueError('Not a valid video url')

    try:
        return loader.load_data(
            ytlinks=[url]
        )
    except youtube_transcript_api._errors.TranscriptsDisabled:
        print(f'No subtitles in "{url}" => no transcript')


if __name__ == '__main__':
    # test_wikipedia_search()
    print(test_yt_trsc())
