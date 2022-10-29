import os
if __name__ == "__main__":
    from dataset.data import myDataloader
    from network.loss import *
    from network.SSIM import SSIM
    from network.Teacher import *
    import numpy as np

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device("cuda")

    class trainer_2:
        def __init__(self, train_gen, step, numD=4):
            super(trainer_2, self).__init__()

            self.numd = numD
            self.step = step
            self.trainloader = train_gen
            self.modelG = endeFUINT2_1().to(device)
            self.optimizer_G = torch.optim.Adam(filter(lambda p: p.requires_grad, self.modelG.parameters()), lr=1e-4,
                                                betas=(0.9, 0.999))
            #self.scheduler_lr = CyclicCosineDecayLR(self.optimizer_G, init_interval=20, min_lr=5e-8, restart_interval=20, restart_lr=1e-4)

            self.criterion = nn.L1Loss().to(device)
            self.VGGLoss = VGGLoss().to(device)
            self.ssim = SSIM().to(device)

            self.weight = [1./4, 1./2, 1]

        def opt_G1(self, Jx, fake):
            self.optimizer_G.zero_grad()

            g_loss_MSE0 = self.criterion(fake, Jx.detach())

            loss = g_loss_MSE0
            loss.backward()

            self.optimizer_G.step()

            return g_loss_MSE0, g_loss_MSE0, g_loss_MSE0

        def train(self, epoch, train_gen):
            self.trainloader = train_gen

            #self.scheduler_lr.step(epoch=epoch-1)
            self.optimizer_G.step()
            print(self.optimizer_G.param_groups[0]["lr"])

            epoch_loss1 = 0
            epoch_loss2 = 0
            epoch_loss3 = 0
            iteration = 0
            for (Ix, Jx) in self.trainloader:
                iteration = iteration+1
                Ix = Ix.to(device)
                Jx = Jx.to(device)

                fake, _ = self.modelG(Jx)

                loss1, loss2, loss3 = self.opt_G1(Jx, fake)

                epoch_loss1 += float(loss1)
                epoch_loss2 += float(loss2)
                epoch_loss3 += float(loss3)

                if iteration % 100 == 0:
                    print(
                        "===> Epoch[{}]({}/{}): Loss{:.4f};".format(epoch, iteration, len(trainloader), loss2.cpu()))

                    Ix_cc = fake# modelD(Detail_I) #+ Ix[:, 6:9, :, :] modelD(Detail_I)#
                    Ix_cc = Ix_cc.clamp(0, 1)
                    Ix_cc = Ix_cc[0].permute(1, 2, 0).detach().cpu().numpy()


                    Ix = Ix.clamp(0, 1)
                    Ix = Ix[0].permute(1, 2, 0).detach().cpu().numpy()
                    Ix_cc = np.hstack([Ix, Ix_cc])

                    Jx = Jx.clamp(0, 1)
                    Jx = Jx[0].permute(1, 2, 0).detach().cpu().numpy()
                    Ix_cc = np.hstack([Ix_cc, Jx])

                    #print(Ix_cc.shape)
                    #imageio.imsave('./results' + '/' + str((epoch - 1) * 100 + iteration / 100) + '.png', np.uint8(Ix_cc*255))

                    print("MSE:{:4f},MSSIM:{:4f},VGG:{:4f},VGG:{:4f}".format(loss1, loss2, loss3, loss1))

            print("===>Epoch{} Complete: Avg loss is :Loss1:{:4f},Loss2:{:4f},Loss3:{:4f},  ".format(epoch, epoch_loss1 / len(trainloader), epoch_loss2 / len(trainloader), epoch_loss3 / len(trainloader)))


    trainloader = myDataloader().getLoader()

    for i in range(1, 2):
        print("===> Loading model and criterion")

        trainModel = trainer_2(trainloader, step=i, numD=1)

        for epoch in range(1, 21):
            print("Step {}:-------------------------------".format(i))
            trainModel.train(epoch, trainloader)
            torch.save(trainModel.modelG, "./models/Teacher/Teacher.pkl")