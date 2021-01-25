# input
print('제목을 입력하세요 : ')
title = input()
print('본문을 입력하세요 : ')
text = input()

from extraction_5H1W import Extraction_5W1H

k = Extraction_5W1H()
k.extract(title, text)
