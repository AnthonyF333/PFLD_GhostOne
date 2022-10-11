# PFLD_GhostOne
&emsp;&emsp;在两年前，我曾经分享过利用GhostNet对PFLD进行优化的文章——[《人脸关键点检测算法PFLD的优化实战记录》](https://blog.csdn.net/u010892804/article/details/108509243?spm=1001.2014.3001.5501)，那里面介绍了经过各种奇技淫巧，GhostNet确实能够提升PFLD的速度和精度，暂且称呼这个方法为PFLD-GhostNet，而且分享到[GitHub：FaceLandmark_PFLD_UltraLight](https://github.com/AnthonyF333/FaceLandmark_PFLD_UltraLight)后获得六十多颗星，也算老怀安慰了。
&emsp;&emsp;两年后，在机缘巧合下接触到Apple大佬的MobileOne，灵机一触觉得MobileOne可能对PFLD-GhostNet还会有进一步的提升，决定尝试一下，便有了今天这篇文章。这次主要记录将GhostNet模块与MobileOne模块进行结合，对PFLD进行进一步优化的实战经验，为需要优化人脸关键点检测算法的小伙伴们提供参考，新的方法我将它叫做PFLD-GhostOne。这个PFLD-GhostOne模型比Slim版本的PFLD-GhostNet精度提升了近3.75%，相比原始PFLD更是提升了4.25%，NCNN推理速度比Slim版本的PFLD-GhostNet提升了超过11%，比原始版本的PFLD提升超过55%。
> * MobileOne简介
> * PFLD-GhostOne优化过程
> * 优化结果

开源代码在以下GitHub链接，欢迎大家多多点星收藏：）
[GitHub：PFLD_GhostOne](https://github.com/AnthonyF333/PFLD_GhostOne)

## MobileOne简介
&emsp;&emsp;有关GhostNet和PFLD的介绍可以参考我两年前分享的文章[《人脸关键点检测算法PFLD的优化实战记录》](https://blog.csdn.net/u010892804/article/details/108509243?spm=1001.2014.3001.5501)，这里主要介绍一下MobileOne。
&emsp;&emsp;MobileOne是2022年Apple大佬提出来的用于移动设备的网络结构，利用重参数化可以有效提高轻量级网络的性能，当然重参数化是RepVGG最先提出来，有兴趣的童鞋可以去膜拜一下，MobileOne只是站在RepVGG巨人的肩膀上提出来的模型结构。如下图所示，MobileOne的基础模块在训练时是多分支的卷积结构，由于卷积和BN都是线性操作，经过重参数化后这个多分支结构可以合并成一个卷积和BN操作，因此在推理阶段，这个多分支结构就可以等效为单通路结构，可以大大减少推理阶段的运算量。在下面的性能表格中可以看到最轻量级的MobileOne-S0比ShuffleNetV2-x1.0的精度已经有一个较大的提升，推理速度两者也相当。

<div align=center>![](https://github.com/AnthonyF333/PFLD_GhostOne/blob/main/img/1.png)

<p align="center"><font size=4.>$\uparrow$ 图1 MobileOne Block结构</font></p>

<div class="center">

![][2]  
</div>
<p align="center"><font size=4.>$\uparrow$ 图2 MobileOne性能表</font></p>

## PFLD-GhostOne优化过程
在这次优化PFLD模型的过程中，最最最重要的部分就是将MobileOne的重参数化技术引入到GhostNet中，我将这个结合体称为GhostOne，正因为这个GhostOne模块，可以大大提升PFLD模型的性能。
### GhostOne模块
在介绍GhostOne之前，我们先重温一下GhostNet的基础结构。
GhostNet的基础结构Ghost Module如图3所示：
<div class="center">

![][5] 
</div>
<p align="center"><font size=4.>$\uparrow$ 图3 Ghost Module</font></p>

通过堆叠Ghost Module形成的Ghost Bottleneck如图4所示：
<div class="center">

![][3] 
</div>
<p align="center"><font size=4.>$\uparrow$ 图4 Ghost Bottleneck</font></p>

现在进入正题，介绍今天的主角GhostOne。
GhostOne Module的整体结构如图5所示：
<div class="center">

![][6]
</div>
<p align="center"><font size=4.>$\uparrow$ 图5 GhostOne Module</font></p>

&emsp;&emsp;可以看到GhostOne Module其实和Ghost Module的整体结构非常相像，两者的最大区别就是GhostOne Module利用MobileOne中的多分支卷积结构代替了Ghost Module中单一的卷积操作。在训练过程中两者的结构可能差异比较大，一旦经过重参数化后，在推理过程中两者的结构理论上是一模一样的，计算量和参数量也都是一样的，因此GhostOne Module对比原始的Ghost Module，在推理速度上是一样的。

通过堆叠GhostOne Module形成的GhostOne Bottleneck如图6所示：
<div class="center">

![][4]
</div>
<p align="center"><font size=4.>$\uparrow$ 图6 GhostOne Bottleneck</font></p>

&emsp;&emsp;通过对比Ghost Bottleneck可以看出，GhostOne Bottleneck缺少了Skip Connection，这里参考的是YoloV7的做法，YoloV7的作者发现，当两个重参数化模块串联时，这个Skip Connection会破坏模型的特征表达能力，最终便有了上面的GhostOne Bottleneck结构。
&emsp;&emsp;最终的PFLD-GhostOne模型结构，就是在PFLD-GhostNet的基础上，直接将上述的GhostOne Bottleneck替换掉原始的Ghost Bottleneck，同时把一般的卷积操作也替换成MobileOne Block，在模型精度有比较大的提升的同时，推理速度也有了一个质的提升。PFLD-GhostOne结构如表1：

<!-- 让表格居中显示的风格 -->
<style>
.center 
{
  width: auto;
  display: table;
  margin-left: auto;
  margin-right: auto;
}
</style>
<div class="center">

Input|Operator|t|c|n|s
:--:|:--:|:--:|:--:|:--:|:--:
112x112x3|MobileOneBlock 3×3|-|64|1|2
56x56x64|DW-MobileOneBlock 3×3|-|64|1|1
56x56x64|GhostOne Bottleneck|1.5|80|3|2
28x28x80|GhostOne Bottleneck|2.5|96|3|2
14x14x96|GhostOne Bottleneck|3.5|144|4|2
7x7x144|GhostOne Bottleneck|1.5|16|1|1
7x7x16|MobileOneBlock 3×3|-|32|1|1
7x7x32|Conv7×7|-|128|1|1
(S1) 56x56x64<br />(S2) 28x28x80<br />(S3) 14x14x96<br />(S4) &ensp;7x7x144<br />(S5) &ensp;1x1x128|AvgPool<br />AvgPool<br />AvgPool<br />AvgPool<br />-|-<br />-<br />-<br />-<br />-|64<br />80<br />96<br />144<br />128|1<br />1<br />1<br />1<br />-|-<br />-<br />-<br />-<br />-
S1,S2,S3,S4,S5|Concat+Full Connection|-|136|1|-
</div>
<p align="center"><font size=4.>$\uparrow$ 表1 PFLD-GhostOne结构</font></p>
说明：t代表GhostOne Bottleneck中间通道的拓展倍数，c代表GhostOne Bottleneck的输出通道数目，n代表GhostOne Bottleneck的串联个数，s代表stride，模型所有的MobileOne Block中的分支数目都是6。

## 优化结果
WFLW测试结果
模型输入大小为112x112
Model|NME|OpenVino Latency(ms)|NCNN Latency(ms)|ONNX Model Size(MB)
:--:|:--:|:--:|:--:|:--:
PFLD|0.05438|1.65(CPU)&emsp;2.78(GPU)|5.4(CPU)&emsp;5.1(GPU)|4.66
PFLD-GhostNet|0.05347|1.79(CPU)&emsp;2.55(GPU)|2.9(CPU)&emsp;5.3(GPU)|3.09
PFLD-GhostNet-Slim|0.05410|2.11(CPU)&emsp;2.54(GPU)|2.7(CPU)&emsp;5.2(GPU)|2.83
PFLD-GhostOne|0.05207|1.79(CPU)&emsp;2.18(GPU)|2.4(CPU)&emsp;5.0(GPU)|2.71
说明：OpenVino和NCNN的推理时间均在11th Gen Intel(R) Core(TM) i5-11500下进行统计。


作者 [@Anthony Github](https://github.com/AnthonyF333)
2022 年 10月
