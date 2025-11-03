#!/bin/bash
# OpenCUA-VL API 테스트 스크립트 (cURL 버전)

API_BASE="http://localhost:8000/v1"
API_KEY="EMPTY"
MODEL="xlangai/OpenCUA-7B"

echo "============================================"
echo "OpenCUA-VL API 테스트 (cURL)"
echo "============================================"
echo "서버: $API_BASE"
echo "모델: $MODEL"
echo ""

# 테스트 1: 텍스트만
echo "============================================"
echo "테스트 1: 텍스트만"
echo "============================================"
curl -X POST "$API_BASE/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d '{
    "model": "'$MODEL'",
    "messages": [
      {
        "role": "user",
        "content": "안녕하세요! OpenCUA-VL 모델을 테스트하고 있습니다. 간단히 자기소개를 해주세요."
      }
    ],
    "max_tokens": 256,
    "temperature": 0.7
  }' | jq -r '.choices[0].message.content'

echo ""
echo ""

# 테스트 2: 이미지 URL 사용
echo "============================================"
echo "테스트 2: 이미지 URL 사용"
echo "============================================"
IMAGE_URL="https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"

curl -X POST "$API_BASE/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d '{
    "model": "'$MODEL'",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "이 이미지를 자세히 설명해주세요."
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "'$IMAGE_URL'"
            }
          }
        ]
      }
    ],
    "max_tokens": 512,
    "temperature": 0.7
  }' | jq -r '.choices[0].message.content'

echo ""
echo ""

# 테스트 3: 로컬 이미지 파일 사용 (base64 인코딩)
if [ -f "$1" ]; then
  echo "============================================"
  echo "테스트 3: 로컬 이미지 파일 사용 ($1)"
  echo "============================================"
  
  # 이미지를 base64로 인코딩
  BASE64_IMAGE=$(base64 -w 0 "$1")
  
  # MIME type 결정
  if [[ "$1" =~ \.(png|PNG)$ ]]; then
    MIME_TYPE="image/png"
  elif [[ "$1" =~ \.(jpg|jpeg|JPG|JPEG)$ ]]; then
    MIME_TYPE="image/jpeg"
  else
    MIME_TYPE="image/jpeg"
  fi
  
  curl -X POST "$API_BASE/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $API_KEY" \
    -d '{
      "model": "'$MODEL'",
      "messages": [
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": "이 이미지에 무엇이 보이나요? 자세히 설명해주세요."
            },
            {
              "type": "image_url",
              "image_url": {
                "url": "data:'$MIME_TYPE';base64,'$BASE64_IMAGE'"
              }
            }
          ]
        }
      ],
      "max_tokens": 512,
      "temperature": 0.7
    }' | jq -r '.choices[0].message.content'
  
  echo ""
  echo ""
fi

echo "테스트 완료!"

