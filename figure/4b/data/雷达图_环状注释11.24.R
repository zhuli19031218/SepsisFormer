# 安装第三方包
# 安装 devtools 包（如果尚未安装）
#install.packages("devtools")
#使用 devtools 从 GitHub 安装 ggradar
#devtools::install_github("ricardo-bion/ggradar", dependencies = TRUE)
#install.packages("ggradar")
# install.packages("tidyverse")
# install.packages("ggplot2")
# install.packages("patchwork")
# install.packages("devtools")
# devtools::install_github("ricardo-bion/ggradar", dependencies = TRUE)
# 导入包
library(ggradar)
library(tidyverse)

# 构造数据：
data <- data.frame(
  # 雷达图中有8条线，5个点
  row.names = LETTERS[1:5],
  `Mild_T` = c(0.216396039,	0,	0.344294004,	0.118288442,	0.12542612),# 每条线中5个点对应的大小 # nolint
  `Moderate_T` = c(0.299187624,	0.302890437,	0.556821298,	0.062637906,	0.345160709), # nolint
  `Severe_T` = c(0.473175447,	0.664350574,	1,	0.430055402,	0.72320442),
  `Dangerous_T` = c(1,	1,	0.800472255,	0.90430622,	1),
  `Mild_C` = c(0,	0.00297527,	0,	0,	0),
  `Moderate_C` = c(0.285239831,	0.288441205,	0.462222151,	0.082399279,	0.304092629), # nolint
  `Severe_C` = c(0.445608005,	0.659462067,	0.876703543,	0.403662665,	0.594229589), # nolint
  `Dangerous_C` = c(0.656302025,	0.944422365,	0.626262626,	1,	0.862983425)
)



# 将行名改为分组列
df <- as.data.frame(t(data)) %>% rownames_to_column("group")
df
#                 group   A   B   C   D   E
# 1   T.cell.activation 1.0 0.3 0.2 0.1 0.1
# 2 Neutrophil.immunity 0.3 1.0 0.2 0.1 0.1
# 3    ECM.organization 0.3 0.2 1.0 0.1 0.1
# 4  Hepatic.metabolism 0.3 0.2 0.1 1.0 0.1
# 5      Cell.migration 0.3 0.2 0.1 0.1 1.0


############## 雷达图 ------------
p2 <- ggradar(
  df,
  axis.labels = rep(NA, 5),# labels个数
  grid.min = 0, grid.mid = 0.5, grid.max = 1,
  # 雷达图线的粗细和颜色：
  group.line.width = 1,# 线的粗细 # nolint
  group.point.size = 2,# 点的大小 # nolint
  group.colours = c("#EABABB","#1F4892","#BA1E2E","#ABC349","#AE9A83","#7620DF","#BB4989","#2C7340"),# 线的颜色
  # 背景边框线颜色：
  background.circle.colour = "white",
  gridline.mid.colour = "#2b8c96",
  legend.position = "none",
  # 不加坐标轴标签：
  label.gridline.min = F,
  label.gridline.mid = F,
  label.gridline.max = F
)+
  theme(plot.background = element_blank(),
        panel.background = element_blank())
p2

############## 圆环注释+文本注释 ------------
library(ggplot2)

# 注释数据：
tmp <- data.frame(x = rep(1, 5),
                  y = rep(1, 5),
                  group = colnames(df)[-1])

# 绘图：
p1 <- ggplot()+
  # 圆环：
  geom_bar(data = tmp, aes(x, y, fill = group), stat = "identity", position = "dodge")+
  # 文本注释：
  geom_text(aes(x = rep(1,25), y = rep(2, 25),
                label = paste0("GENE", LETTERS[1:25]), group = 1:25),
            color = "black", size = 2.5,
            position = position_dodge(width = 0.9))+
  # geom_text(aes(x, y, label = gsub("[.]", " ", df$group), group = group),
  #           color = "white",
  #           position = position_dodge(width = 0.9))+
  scale_fill_manual(values = c("#ae9a83", "#804596", "#4cb46d", "#34485c", "#dc713c"))+
  ylim(-5.5,2)+
  # 0.63是计算得来，5个色块，第一个色块的正中心要对准0的位置，
  # 所以2pi/10=0.628即为第一个色块左边界的位置
  coord_polar(start = -0.63)+
  theme_void()+
  theme(legend.position = "none")

p1



######## 拼图 ---------------------
library(patchwork)

p1 + inset_element(p2, left = 0, bottom = 0, right = 0.99, top = 0.99)
# 保存图片为pdf
ggsave("E:/figure/4b/雷达图_环状注释.pdf", height = 5, width = 5)