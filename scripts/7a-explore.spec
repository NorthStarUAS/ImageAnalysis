# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['7a-explore.py'],
             pathex=['H:\\Projects\\ImageAnalysis\\scripts'],
             binaries=[
                 ('C:\\Users\curt\Anaconda3\Lib\site-packages\panda3d\cgGL.dll', '.'),
                 ('C:\\Users\curt\Anaconda3\Lib\site-packages\panda3d\libpandagl.dll', '.'),
                 ('C:\\Users\curt\Anaconda3\Lib\site-packages\panda3d\libp3assimp.dll', '.'),
                 ('C:\\Users\curt\Anaconda3\Lib\site-packages\panda3d\libp3ptloader.dll', '.'),
                 ('C:\\Users\curt\Anaconda3\Lib\site-packages\panda3d\libp3windisplay.dll', '.')
             ],
             datas=[
                 ('C:\\Users\curt\Anaconda3\Lib\site-packages\panda3d\etc\*.prc', 'etc'),
                 ('explore/*.png', 'explore')
             ],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='7a-explore',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='7a-explore')
