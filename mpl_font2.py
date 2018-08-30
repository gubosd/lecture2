import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

prop=fm.FontProperties(fname='C:/Windows/Fonts/Daum_Regular.ttf')
#prop=fm.FontProperties('Daum')

plt.subplot(1,2,1)
plt.plot([1,2,3,4])
plt.title('헬로 플롯! (Hello, plot!)',fontproperties=prop,fontsize=20)
plt.ylabel('Y축입니다',fontproperties=prop)
plt.xticks(range(4),['영','일','이','삼'],fontproperties=prop,rotation=0,fontsize=15)
plt.yticks(range(5))
plt.text(1,3,'안녕하세요',fontproperties=prop,color='g',fontsize=20)
plt.grid(linestyle=':')
plt.axis('equal')

plt.subplot(1,2,2)
plt.title('두번째 플롯',fontproperties=prop)
plt.scatter([1,2,3],[3,4,5])

plt.show()