# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['explorer.py'],
             pathex=['Z:\\Projects\\ImageAnalysis\\scripts'],
             binaries=[
                 ('C:\\Users\curt\Anaconda3\*.dll', '.'),
                 ('C:\\Users\curt\Anaconda3\Lib\site-packages\panda3d\*.dll', '.')
             ],
             datas=[
                 ('C:\\Users\curt\Anaconda3\Lib\site-packages\panda3d\etc\*.prc', 'etc'),
                 ('explore/*.png', 'explore'),
                 ('explore/*.frag', 'explore'),
                 ('explore/*.vert', 'explore')
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
          name='explorer',
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
               name='explorer')
