from hezarfen_ai.model import ModelLoader

if __name__ == "__main__":
    model_loader = ModelLoader()
    model_loader.run_model()

    while True:
        text = input("Metni Giriniz: ")
        result = model_loader.ask(text)
        print("Sonuç: Gerçek" if result[0] == "TRUE" else "Sonuç: Sahte")