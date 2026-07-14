from src.dataset.dataset_loader import DatasetLoader
def main():
    loader = DatasetLoader("data/images")
    images_metadata = loader.load_images()
    print(images_metadata[:3])
if __name__ == "__main__":
    main()