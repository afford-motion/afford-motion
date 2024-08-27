from smplkit import SMPLXLayer

class BaseDataset():
    def __init__(self, data_dir: str) -> None:
        self.data_dir = data_dir
        self.device = "cpu"
        
        self.num_pca_comps = 12
        self.body_model_male = SMPLXLayer(
            gender='male', num_pca_comps=self.num_pca_comps).to(device=self.device)
        self.body_model_female = SMPLXLayer(
            gender='female', num_pca_comps=self.num_pca_comps).to(device=self.device)
        self.body_model_neutral = SMPLXLayer(
            gender='neutral', num_pca_comps=self.num_pca_comps).to(device=self.device)

    def process(self) -> None:
        pass
