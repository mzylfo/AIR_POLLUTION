

            for batch_num, dataBatch in enumerate(dataBatches):
                ###########################
                # 1  Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                
                ###########################
                ##  A Update D with real data
                ###########################
                self.model_dis.zero_grad()
                self.optimizer_dis.zero_grad()
                batch_real_err_D = torch.zeros(1).to(device=self.device) 
                for i, item in enumerate(dataBatch):
                    samplef = item['sample']
                    sample = samplef.type(torch.float32)
                    label = torch.full((1,), real_label, dtype=torch.float32).to(device=self.device)      
                    output = self.model_dis(sample)['x_output'].view(-1)                    
                    item_err = self.criterion(output, label)
                    batch_real_err_D += item_err
                err_D_r_list.append(batch_real_err_D.detach().cpu().numpy()[0]/len(dataBatch))
                batch_real_err_D.backward()
                ##self.optimizer_dis.step()
                
                
                ###########################
                ##  make noiseData
                ###########################
                noise_batch = list()
                for i, item in enumerate(dataBatch):
                    noise = torch.randn(1, 1, noise_size[0], noise_size[1]).to(device=self.device) 
                    noise_batch.append(noise)
                
                ###########################
                ##  B Update D with fake data
                ###########################
                ##self.optimizer_dis.zero_grad()
                batch_fake_err_D = torch.zeros(1).to(device=self.device) 
                for i, item in enumerate(noise_batch):
                    fake = self.model_gen(item)['x_output'] 
                    label = torch.full((1,), fake_label, dtype=torch.float32).to(device=self.device)                    
                    output = self.model_dis(fake.detach())['x_output'].view(-1)
                    item_err = self.criterion(output, label)
                    batch_fake_err_D += item_err
                err_D_f_list.append(batch_fake_err_D.detach().cpu().numpy()[0]/len(dataBatch))
                
                batch_fake_err_D.backward()
                self.optimizer_dis.step() 
                
                ###########################               
                errD = batch_real_err_D + batch_fake_err_D
                err_D_list.append(errD.detach().cpu().numpy())
                
                ###########################
                # 2 Update G network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                A=list(self.model_gen.parameters())[-1].clone()
                
                self.optimizer_gen.zero_grad()
                batch_fake_err_G = torch.zeros(1).to(device=self.device) 
                self.model_gen.train()

                for i, item in enumerate(noise_batch):
                    fake = self.model_gen(item)['x_output']                    
                    label = torch.full((1,), real_label, dtype=torch.float32).to(device=self.device) 
                    output = self.model_dis(fake)['x_output'].view(-1)                    
                    item_err = self.criterion(output, label)
                    batch_fake_err_G += item_err 
                batch_fake_err_G.backward()
                self.optimizer_gen.step() 
                err_G_list.append(batch_fake_err_G.detach().cpu().numpy()[0]/len(noise_batch))
                  
                errG = batch_fake_err_G
            -
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            -----------
            
            
            
            for i, item in enumerate(dataBatch)
                samplef = item['sample']
                sample = samplef.type(torch.float32)
                
                ###########################
                # 1  Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
            
                ###########################
                ##  A Update D with real data
                ###########################
                self.model_dis.zero_grad()
                self.optimizer_dis.zero_grad()               
                
                label = torch.full((1,), real_label, dtype=torch.float32).to(device=self.device)      
                output = self.model_dis(sample)['x_output'].view(-1)                    
                item_err = self.criterion(output, label)
                item_err.backward()
                self.optimizer_dis.step()
                
                ###########################
                ##  B Update D with fake data
                ###########################
                self.optimizer_dis.zero_grad()                
                noise = torch.randn(1, 1, noise_size[0], noise_size[1]).to(device=self.device) 
                fake = self.model_gen(noise)['x_output'] 
                label = torch.full((1,), fake_label, dtype=torch.float32).to(device=self.device)                    
                output = self.model_dis(fake.detach())['x_output'].view(-1)
                item_err = self.criterion(output, label)
                item_err.backward()
                self.optimizer_dis.step() 
            
                ###########################
                # 2 Update G network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                self.optimizer_gen.zero_grad()
                noise = torch.randn(1, 1, noise_size[0], noise_size[1]).to(device=self.device) 
                fake = self.model_gen(noise)['x_output']                    
                
                if first:
                    print("NOISE: (epoch",epoch,")\t",noise)
                    print("")
                    print("FAKE: (epoch",epoch,")\t",fake)
                    print("================================================================================================================================================\n\n")
                    first = False
                    
                label = torch.full((1,), real_label, dtype=torch.float32).to(device=self.device) 
                output = self.model_dis(fake)['x_output'].view(-1)                    
                item_err = self.criterion(output, label)
                item_err.backward()
                self.optimizer_gen.step() 