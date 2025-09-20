import time
import random
from concurrent import futures

import grpc

import sentiment_pb2 as pb
import sentiment_pb2_grpc as pb_grpc


class SentimentService(pb_grpc.SentimentServiceServicer):
    def Analyze(self, request: pb.AnalyzeRequest, context) -> pb.AnalyzeResponse:
        labels = ["positive", "negative", "neutral"]
        dominant = random.choice(labels)

        scores = [
            pb.SentimentScore(label=l, score=random.random())
            for l in labels
        ]
        return pb.AnalyzeResponse(
            text=request.text,
            dominant_label=dominant,
            scores=scores,
        )

    def BatchAnalyze(self, request: pb.BatchAnalyzeRequest, context) -> pb.BatchAnalyzeResponse:
        results = []
        for item in request.items:
            labels = ["positive", "negative", "neutral"]
            dominant = random.choice(labels)

            scores = [
                pb.SentimentScore(label=l, score=random.random())
                for l in labels
            ]
            results.append(
                pb.AnalyzeResponse(
                    text=item.text,
                    dominant_label=dominant,
                    scores=scores,
                )
            )
        return pb.BatchAnalyzeResponse(results=results)


def serve(host: str = "127.0.0.1", port: int = 50051):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    pb_grpc.add_SentimentServiceServicer_to_server(SentimentService(), server)
    server.add_insecure_port(f"{host}:{port}")
    server.start()
    print(f"🚀 Fake Sentiment gRPC server running on {host}:{port}")
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        print("Shutting down...")
        server.stop(0)


if __name__ == "__main__":
    serve()
