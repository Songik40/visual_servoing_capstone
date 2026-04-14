#!/usr/bin/env python3
import speech_recognition as sr

def test_airpods_mic():
    # 음성 인식기 생성
    r = sr.Recognizer()
    
    # 시스템의 기본 마이크(현재 우분투에서 선택된 마이크)를 사용
    with sr.Microphone() as source:
        print("========================================")
        print("에어팟 마이크 세팅 중... (주변 소음 분석 1초)")
        # 주변 소음을 측정해서 노이즈 캔슬링 기준을 잡습니다. (필수!)
        r.adjust_for_ambient_noise(source, duration=1)
        
        print("듣는 중...")
        
        try:
            # 마이크에서 소리를 듣고 메모리에 저장 (최대 5초 대기, 최대 10초 녹음)
            audio = r.listen(source, timeout=5, phrase_time_limit=10)
            print("오디오 수신 완료. 텍스트로 변환 중...")
            
            # 구글의 무료 음성 인식 엔진을 사용하여 한국어(ko-KR)로 변환
            text = r.recognize_google(audio, language='ko-KR')
            print("========================================")
            print(f"입력된 문장: '{text}'")
            print("========================================")
            
        except sr.WaitTimeoutError:
            print("error: 지정된 시간 동안 아무 소리도 들리지 않았습니다.")
        except sr.UnknownValueError:
            print("error: 소리는 들렸지만 무슨 말인지 정확히 알아듣지 못했습니다.")
        except sr.RequestError as e:
            print(f"error: 구글 서버 연결에 실패했습니다. (인터넷 확인) {e}")

if __name__ == "__main__":
    test_airpods_mic()
