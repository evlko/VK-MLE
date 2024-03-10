PROJECT="VK MLE Simple RecSys API"

run:
	uvicorn app.app:app --reload

tensorboard:
	tensorboard --logdir=model/tb_logs --bind_all

pretty:
	black .
	isort .

.PHONY: run