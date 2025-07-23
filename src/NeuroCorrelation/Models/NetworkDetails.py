from torchsummary import summary
from pathlib import Path

class NetworkDetails():
    
    def __init__(self, model, loss, path, edge_index=None):
        self.model = model
        self.loss = loss
        self.path = path
        self.edge_index = edge_index
        
    def saveModelParams(self):
        file_str = ""
        for key in self.model:
           
            ann_str = self.saveModel_structure(key, self.model[key])
            if key =="AE":                
                loss_str = self.saveModel_loss()
            else:
                loss_str = "----"
            file_str_key = f"================ {key} ================\n======== structure\n{ann_str}\n\n======== loss\n{loss_str}"
            file_str = "\n".join([file_str, file_str_key])
        filename = Path(self.path, "summary_network.txt")
        with open(filename, 'w') as file:
            file.write(file_str)
        print("SETTING PHASE: Summary model file - DONE")
    
    def saveModel_structure(self, key, model_net):
        net_summary = model_net.summary()
        
        return f"{key}::\n {net_summary}"
    
    def saveModel_loss(self):
        loss_str = ""
        for key in self.loss:
            loss_summary = self.loss[key].get_Loss_params()            
            loss_summary_str = f"{key}:: loss: {loss_summary}"
            loss_str = '\n'.join([loss_str, loss_summary_str])
        return loss_str