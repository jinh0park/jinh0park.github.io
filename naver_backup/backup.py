from naverblogbacker.utils import isEmptyDirectory
from naverblogbacker.blog import BlogCrawler
from naverblogbacker.utils import getTotalCount
import os

myId = 'jinh0park'
myPath = './posts'
mySkipSticker = True

print(getTotalCount(myId))

if isEmptyDirectory(dirPath=myPath):
    myBlog = BlogCrawler(
        targetId=myId, skipSticker=mySkipSticker, isDevMode=False)
    myBlog.crawling(dirPath=myPath)

    # 정상적으로 실행 시 백업 후 에러 포스트 개수가 출력
    print(
        f'[MESSAGE] Complete! your blog posts, the number of error posts is {BlogCrawler.errorPost}')
    # 위의 메세지를 잘 보기 위해 프로그램 종료 전 정지
    os.system("pause")
