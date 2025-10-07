from pydoc import text
import grpc

import server_py.sentiment_pb2 as pb
import server_py.sentiment_pb2_grpc as pb_grpc


texts = [
    ("A MOL vezetője nyilatkozott a sajtónak Debrecenben.", 'hun'),
    ("Ferenc pápa misét tartott a Hősök terén.", 'hun'),
    ("Kovács Anna elutazott Párizsba a barátaival.", 'hun'),
    ("A Microsoft új irodát nyitott Szegeden.", 'hun'),
    ("Gyurcsány Ferenc beszédet mondott az Európai Parlamentben.", 'hun'),
    ("Az OTP Bank támogatást nyújtott a Miskolci Egyetemnek.", 'hun'),
    ("Szabó Péter és Nagy Katalin közösen szerveztek konferenciát Berlinben.", 'hun'),
    ("Putyin és Zelenszkij találkoztak Isztambulban.", 'hun'),
    ("A Real Madrid legyőzte a Ferencvárost Budapesten.", 'hun')
]


def run():
    with grpc.insecure_channel("127.0.0.1:50051") as channel:
        stub = pb_grpc.SentimentServiceStub(channel)
        for item in texts:
            text, lang = item
            response = stub.Analyze(pb.AnalyzeRequest(text=text, language=lang))
            print(response)


if __name__ == "__main__":
    run()