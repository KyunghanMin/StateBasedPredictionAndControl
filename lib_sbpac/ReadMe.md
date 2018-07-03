## Library package for SbPaC

누구나 들어와서 사용할 python library들을 만들어 주엇

## 설치방법
conda environment activation 후, 

```conda
conda install -c kyunghan lib_sbpac
```

## 사용방법
spyder에서
```python
#Import module
import lib_sbpac

#Load color set
colorset_variable = lib_sbpac.color_code.get_ColorSet()

#Display color set
lib_sbpac.color_code.dis_ColorSet()
```

## Description

Version: 0.0.9

Platform: python 3.5

List: 

* color_code
  * Fcn: get_ColorSet()
  * Fcn: dis_ColorSet()

*물론 git으로 release and build 환경은 구축하지 않았기 때문에 build가 필요할때는 kyunghan 에게 문의*
