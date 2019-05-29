import sys
sys.path.append('../')
from pycore.tikzeng import *
from pycore.blocks  import *
    # defined your arch
arch = [
    to_head( '../' ),
    to_cor(),
    to_begin(),

    to_input( 'taihu_on_flowchart.jpg',height=14, width=14),


    to_Conv("e1c1", 256, 64, offset="(0,0,0)", to="(0,0,0)", height=64, depth=64, width=2 ),
    to_Conv("e1c2", 256, 64, offset="(0,0,0)", to="(e1c1-east)", height=64, depth=64, width=2 ),
    to_Pool("e1p1", offset="(0,0,0)", to="(e1c2-east)", height=48, depth=48, width=1),
    #encoder2
    to_Conv("e2c1", 128, 128, offset="(3,0,0)", to="(e1p1-east)", height=48, depth=48, width=4 ),
    to_connection( "e1p1", "e2c1"),
    to_Conv("e2c2", 128, 128, offset="(0,0,0)", to="(e2c1-east)", height=48, depth=48, width=4),
    to_Pool("e2p1", offset="(0,0,0)", to="(e2c2-east)", height=36, depth=36, width=1),
    
    #encoder3
    to_Conv("e3c1", 64, 256, offset="(2,0,0)", to="(e2p1-east)", height=36, depth=36, width=6 ),
    to_connection( "e2p1", "e3c1"),
    to_Conv("e3c2", 64, 256, offset="(0,0,0)", to="(e3c1-east)", height=36, depth=36, width=6),
    to_Conv("e3c3", 64, 256, offset="(0,0,0)", to="(e3c2-east)", height=36, depth=36, width=6),
    to_Pool("e3p1", offset="(0,0,0)", to="(e3c3-east)", height=28, depth=28, width=1),
    #encoder4
    to_Conv("e4c1", 32, 512, offset="(1.5,0,0)", to="(e3p1-east)", height=28, depth=28, width=8 ),
    to_connection( "e3p1", "e4c1"),
    to_Conv("e4c2", 32, 512, offset="(0,0,0)", to="(e4c1-east)", height=28, depth=28, width=8),
    to_Conv("e4c3", 32, 512, offset="(0,0,0)", to="(e4c2-east)", height=28, depth=28, width=8),
    to_Pool("e4p1", offset="(0,0,0)", to="(e4c3-east)", height=20, depth=20, width=1),
    #encoder5
    to_Conv("e5c1", 16, 1024, offset="(1.5,0,0)", to="(e4p1-east)", height=20, depth=20, width=10 ),
    to_connection( "e4p1", "e5c1"),
    to_Conv("e5c2", 16, 1024, offset="(0,0,0)", to="(e5c1-east)", height=20, depth=20, width=10),
    to_Conv("e5c3", 16, 1024, offset="(0,0,0)", to="(e5c2-east)", height=20, depth=20, width=10),
    to_Pool("e5p1", offset="(0,0,0)", to="(e5c3-east)", height=15, depth=15, width=1),
    #decoder1
    to_UnPool("d1p1", offset="(1,0,0)", to="(e5p1-east)", height=20, depth=20, width=1),
    to_connection( "e5p1", "d1p1"),
    to_ConvRes("d1c1", 16, 1024, offset="(0,0,0)", to="(d1p1-east)", height=20, depth=20, width=10),
    to_Conv("d1c2", 16, 1024, offset="(0,0,0)", to="(d1c1-east)", height=20, depth=20, width=10),
    to_ConvRes("d1c3", 16, 1024, offset="(0,0,0)", to="(d1c2-east)", height=20, depth=20, width=10),
    #decoder2
    to_UnPool("d2p1", offset="(1.5,0,0)", to="(d1c3-east)", height=28, depth=28, width=1),
    to_connection( "d1c3", "d2p1"),
    to_ConvRes("d2c1", 32, 512, offset="(0,0,0)", to="(d2p1-east)", height=28, depth=28, width=8),
    to_Conv("d2c2", 32, 512, offset="(0,0,0)", to="(d2c1-east)", height=28, depth=28, width=8),
    to_ConvRes("d2c3", 32, 512, offset="(0,0,0)", to="(d2c2-east)", height=28, depth=28, width=8),
    #decoder3
    to_UnPool("d3p1", offset="(2,0,0)", to="(d2c3-east)", height=36, depth=36, width=1),
    to_connection( "d2c3", "d3p1"),
    to_ConvRes("d3c1", 64, 256, offset="(0,0,0)", to="(d3p1-east)", height=36, depth=36, width=6),
    to_Conv("d3c2", 64, 256, offset="(0,0,0)", to="(d3c1-east)", height=36, depth=36, width=6),
    to_ConvRes("d3c3", 64, 256, offset="(0,0,0)", to="(d3c2-east)", height=36, depth=36, width=6),
    #decoder4
    to_UnPool("d4p1", offset="(2,0,0)", to="(d3c3-east)", height=48, depth=48, width=1),
    to_connection( "d3c3", "d4p1"),
    to_ConvRes("d4c1", 64, 256, offset="(0,0,0)", to="(d4p1-east)", height=48, depth=48, width=4),
    to_Conv("d4c2", 64, 256, offset="(0,0,0)", to="(d4c1-east)", height=48, depth=48, width=4),
    #decoder5
    to_UnPool("d5p1", offset="(2,0,0)", to="(d4c2-east)", height=64, depth=64, width=1),
    to_connection( "d4c2", "d5p1"),
    to_ConvRes("d5c1", 64, 256, offset="(0,0,0)", to="(d5p1-east)", height=64, depth=64, width=2),
    to_Conv("d5c2", 64, 256, offset="(0,0,0)", to="(d5c1-east)", height=64, depth=64, width=2),
    


    to_SoftMax("soft1", 10 ,"(3,0,0)", "(d5c2-east)", caption="SOFT",height=64, depth=64, width=1  ),
    to_connection("d5c2", "soft1"),
    to_end()
    ]
def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()
