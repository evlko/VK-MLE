PROJECT="VK MLE Simple RecSys API"

run:
	uvicorn app.app:app --reload

.PHONY: run