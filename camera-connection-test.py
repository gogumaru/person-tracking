from onvif import ONVIFCamera

# Technical Scaffold: Query the ONVIF service for the RTSP path
def get_rtsp_uri():
    try:
        # Tuya/Smart Life usually uses port 8000 for ONVIF
        mycam = ONVIFCamera('10.167.170.4', 8000, 'admin', '696969420')
        
        # Create media service
        media = mycam.create_media_service()
        
        # Get profiles
        profiles = media.GetProfiles()
        token = profiles[0].token
        
        # Get Stream URI
        obj = media.create_type('GetStreamUri')
        obj.StreamSetup = {'Stream': 'RTP-Unicast', 'Transport': {'Protocol': 'RTSP'}}
        obj.ProfileToken = token
        res = media.GetStreamUri(obj)
        
        print(f"Working RTSP URI: {res.Uri}")
    except Exception as e:
        print(f"Discovery Failed: {e}")

if __name__ == "__main__":
    get_rtsp_uri()
