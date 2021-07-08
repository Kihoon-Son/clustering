# clustering test code with fake data.

###*일단, 총 6가지 정보 중 4가지 정보만 사용한 결과입니다. 나머지 2 정보는 완성되는대로 바로 업로드 하겠습니다.

#### - classification_result: 클러스터링 결과가 저장되는 곳
#### - data/results/: 전체 도면 이미지가 저장되어 있는 곳
#### - data/all_fp.json: 각 평면도들의 4가지 정보 number of rooms (nr); floor shiloutte (fs); room location (rl); room connectivity (rc)를 모아놓은 json 파일.

#### - sim_result.csv: 어느 한 도면에 대한 유사도 결과값들을 정리해 놓은 csv파일 (gmatch4py가 작동하지 않아 리눅스에서 만들어온 파일)
#### - test.py: 무시하셔도 됩니다.
#### - main.py: MeanShift와 K-means를 테스트함.
#### - utils.py: 유사도 계산 함수들이 있는 파일.
