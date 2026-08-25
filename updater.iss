[Setup]
AppName=Tomato Analytics Updater
AppVersion=1.1
; This points to where the original app was installed on your friend's PC
DefaultDirName={userappdata}\TomatoAnalytics
OutputBaseFilename=TomatoAnalytics_Model_Update
Compression=lzma2
SolidCompression=yes
PrivilegesRequired=lowest
OutputDir=Output

[Files]
; This takes the newly trained model from your repo and overwrites the old one on his PC
Source: "runs\detect\tomato_blossom_model\weights\best.pt"; DestDir: "{app}\_internal\runs\detect\train\weights"; Flags: ignoreversion
