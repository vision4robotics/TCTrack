import torch.nn as nn

   
import torch as t
import torch.nn.functional as F


class CondConv2d(nn.Module):
    

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 num_experts=1):
        super(CondConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.num_experts = num_experts

        self.avgpool = nn.AdaptiveAvgPool3d((None,1,1))
        self.temporalconv=nn.Conv3d(in_channels, in_channels, (3,1,1))
        self.fc=nn.Conv3d(in_channels, 1, (3,1,1))


        self.weight = nn.Parameter(
            t.Tensor(1,1,out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(t.Tensor(1,1,out_channels))
        else:
            self.register_parameter('bias', None)
        
        for m in self.modules():
            if isinstance(m,nn.Conv3d):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)
			
    def generateweight(self,xet):
	    
	
        xet=xet.permute(0,2,1,3,4)  #x BxCxLxHxW
        xet=self.avgpool(xet) #x BxCxLx1x1
        
        
        allxet=t.cat((xet[:,:,0,:,:].unsqueeze(2),xet[:,:,0,:,:].unsqueeze(2),xet),2)
        calibration=self.temporalconv(allxet)
        
        finalweight=self.weight*(calibration+1).unsqueeze(0).permute(1,3,0,2,4,5)
		
        bias=self.bias*(self.fc(allxet)+1).squeeze().unsqueeze(-1)
        
        
        return finalweight,bias,allxet
    
    def initset(self,x):
        finalweight,finalbias,featset=self.generateweight(x)
        
        b,l, c_in, h, w = x.size()

        x=x.reshape(1,-1,h,w)
        finalweight=finalweight.reshape(-1,self.in_channels,self.kernel_size ,self.kernel_size )
        finalbias=finalbias.view(-1)

        if self.bias is not None:
            
            output = F.conv2d(
                x, weight=finalweight, bias=finalbias, stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=b*l)
        else:
            output = F.conv2d(
                x, weight=finalweight, bias=None, stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=b*l)
        
        output = output.view(-1, self.out_channels, output.size(-2), output.size(-1))
        
        
        return output,featset
    
    
    def combinefeat(self,xet,feat):
        xet=xet.permute(0,2,1,3,4)  #x BxCxLxHxW
        xet=self.avgpool(xet) #x BxCxLx1x1
        
        
        allxet=t.cat((feat[:,:,-2,:,:].unsqueeze(2),feat[:,:,-1,:,:].unsqueeze(2),xet),2)
        calibration=self.temporalconv(allxet)
        
        finalweight=self.weight*(calibration+1).unsqueeze(0).permute(1,3,0,2,4,5)
		
        bias=self.bias*(self.fc(allxet)+1).squeeze().unsqueeze(-1)
        
        

        return finalweight,bias,allxet
        
    def conti(self,x,feat):
        
        finalweight,finalbias,allxet=self.combinefeat(x,feat)
        
        b,l, c_in, h, w = x.size()
        
        
        x=x.reshape(1,-1,h,w)
        finalweight=finalweight.reshape(-1,self.in_channels,self.kernel_size ,self.kernel_size )
        finalbias=finalbias.view(-1)
	
        if self.bias is not None:
            
            output = F.conv2d(
                x, weight=finalweight, bias=finalbias, stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=b*l)
        else:
            output = F.conv2d(
                x, weight=finalweight, bias=None, stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=b*l)
        
        output = output.view(-1, self.out_channels, output.size(-2), output.size(-1))
       

        return output,allxet
    
    def forward(self, x): #x B*L*C*W*H
	
        finalweight,finalbias,_=self.generateweight(x)
        
        b,l, c_in, h, w = x.size()
        
        
        x=x.reshape(1,-1,h,w)
        finalweight=finalweight.reshape(-1,self.in_channels,self.kernel_size ,self.kernel_size )
        finalbias=finalbias.view(-1)
        		
        
        		
        if self.bias is not None:
            
            output = F.conv2d(
                x, weight=finalweight, bias=finalbias, stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=b*l)
        else:
            output = F.conv2d(
                x, weight=finalweight, bias=None, stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=b*l)
        
        output = output.view(-1, self.out_channels, output.size(-2), output.size(-1))
        return output
		
class TemporalAlexNet(nn.Module):
    configs = [3, 96, 256, 384, 384, 256]
	
	#input (B*L)*C*W*H, A1,A2,A3,A4,B1,B2,B3,B4...

    def __init__(self, width_mult=1):
        configs = list(map(lambda x: 3 if x == 3 else
                       int(x*width_mult), TemporalAlexNet.configs))
        super(TemporalAlexNet, self).__init__()
        
        self.block1 = nn.Sequential(
            nn.Conv2d(configs[0], configs[1], kernel_size=11, stride=2),
            nn.BatchNorm2d(configs[1]),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            )
        self.block2 = nn.Sequential(
            nn.Conv2d(configs[1], configs[2], kernel_size=5),
            nn.BatchNorm2d(configs[2]),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            )
        self.block3 = nn.Sequential(
            nn.Conv2d(configs[2], configs[3], kernel_size=3),
            nn.BatchNorm2d(configs[3]),
            nn.ReLU(inplace=True),
            )
        self.temporalconv1 = CondConv2d(configs[3], configs[4], kernel_size=3)
        
        self.b_f1=  nn.Sequential(
            nn.BatchNorm2d(configs[4]),
            nn.ReLU(inplace=True))

        self.temporalconv2 =CondConv2d(configs[4], configs[5], kernel_size=3)
        
        self.b_f2= nn.BatchNorm2d(configs[5])
            
        self.feature_size = configs[5]
        for param in self.block1.parameters():
                param.requires_grad = False
        for param in self.block2.parameters():
                param.requires_grad = False
                
    def init(self, xset):

        		

        xset = self.block1(xset)
        
        xset = self.block2(xset)
        
        xset = self.block3(xset)
		
        xset=xset.unsqueeze(1)
        xset,feat1 = self.temporalconv1.initset(xset)
        
        xset = self.b_f1(xset)
        
        
        xset=xset.unsqueeze(1)
        xset,feat2 = self.temporalconv2.initset(xset)
        
        xset = self.b_f2(xset)		
        		
        return xset,feat1,feat2	
    
    def eachtest(self, xset,feat1,feat2):


        xset = self.block1(xset)
        
        xset = self.block2(xset)
        
        xset = self.block3(xset)
		
        xset=xset.unsqueeze(1)
        xset,feat1 = self.temporalconv1.conti(xset,feat1)
        xset = self.b_f1(xset)
        
        xset=xset.unsqueeze(1)
        xset,feat2 = self.temporalconv2.conti(xset,feat2)
        xset = self.b_f2(xset)		
        		
        return xset,feat1,feat2	
    

    def forward(self, xset):
        B,L, _,_,_ = xset.size()
        		

        
        xset=xset.view(-1,xset.size(-3),xset.size(-2),xset.size(-1))
        xset = self.block1(xset)
        
        xset = self.block2(xset)
        
        xset = self.block3(xset)
		
        xset=xset.view(B,L,xset.size(-3),xset.size(-2),xset.size(-1))
        xset = self.temporalconv1(xset)
        xset = self.b_f1(xset)
        
        
        xset=xset.view(B,L,xset.size(-3),xset.size(-2),xset.size(-1))
        xset = self.temporalconv2(xset)
        xset = self.b_f2(xset)
        		
        		
        return xset	
	
		