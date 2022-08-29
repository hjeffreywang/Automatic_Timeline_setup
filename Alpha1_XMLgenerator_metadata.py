#!/usr/bin/env python
# coding: utf-8

# # Part 4: XML creation
# ## Dependencies: OTIO and pandas
# 
# 

# In[1]:


#audio
from __future__ import print_function
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
import librosa
import librosa.display


# In[2]:


import pandas as pd 
import pyloudnorm as pyln


# In[3]:


#video
import opentimelineio as otio


# In[4]:


import cv2

def dataframe_getintervals(series,desiredvalue):
    #make sure series is the df['column']
    t=series.index[series==desiredvalue].to_series()
    interval_list=t.groupby(t.diff().ne(1).cumsum()).agg(['first','last']).apply(tuple,1).tolist()
    
    return interval_list


from filepaths import *




# In[6]:


example_metadata={'fcp_xml': {'@AM.TrackScrollPosition': '0', '@AM.TrackVScrollPosition': '75', '@MZ.EditLine': '6600425832000', '@MZ.Sequence.AudioTimeDisplayFormat': '200', '@MZ.Sequence.EditingModeGUID': '795454d9-d3c2-429d-9474-923ab13b7018', '@MZ.Sequence.PreviewFrameSizeHeight': '1080', '@MZ.Sequence.PreviewFrameSizeWidth': '1920', '@MZ.Sequence.PreviewRenderingClassID': '1061109567', '@MZ.Sequence.PreviewRenderingPresetCodec': '1297107278', '@MZ.Sequence.PreviewRenderingPresetPath': 'EncoderPresets/SequencePreview/795454d9-d3c2-429d-9474-923ab13b7018/I-Frame Only MPEG.epr', '@MZ.Sequence.PreviewUseMaxBitDepth': 'false', '@MZ.Sequence.PreviewUseMaxRenderQuality': 'false', '@MZ.Sequence.VideoTimeDisplayFormat': '110', '@Monitor.ProgramZoomIn': '0', '@Monitor.ProgramZoomOut': '279167288400000', '@TL.SQAVDividerPosition': '0.5', '@TL.SQAudioVisibleBase': '0', '@TL.SQHeaderWidth': '236', '@TL.SQHideShyTracks': '0', '@TL.SQTimePerPixel': '0.19999999999999998', '@TL.SQVideoVisibleBase': '0', '@TL.SQVisibleBaseTime': '0', '@explodedTracks': 'true', '@id': 'sequence-2', 'labels': {'label2': 'Forest'}, 'logginginfo': {'description': None, 'good': None, 'lognote': None, 'originalaudiofilename': None, 'originalvideofilename': None, 'scene': None, 'shottake': None}, 'media': {'audio': {'format': {'samplecharacteristics': {'depth': '16', 'samplerate': '48000'}}, 'numOutputChannels': '2', 'outputs': {'group': [{'channel': {'index': '1'}, 'downmix': '0', 'index': '1', 'numchannels': '1'}, {'channel': {'index': '2'}, 'downmix': '0', 'index': '2', 'numchannels': '1'}]}}, 'video': {'format': {'samplecharacteristics': {'anamorphic': 'FALSE', 'codec': {'appspecificdata': {'appmanufacturer': 'Apple Inc.', 'appname': 'Final Cut Pro', 'appversion': '7.0', 'data': {'qtcodec': {'codecname': 'Apple ProRes 422', 'codectypecode': 'apcn', 'codectypename': 'Apple ProRes 422', 'codecvendorcode': 'appl', 'datarate': '0', 'keyframerate': '0', 'spatialquality': '1024', 'temporalquality': '0'}}}, 'name': 'Apple ProRes 422'}, 'colordepth': '24', 'fielddominance': 'none', 'height': '2160', 'pixelaspectratio': 'square', 'rate': {'ntsc': 'TRUE', 'timebase': '29.97'}, 'width': '3840'}}}}, 'rate': {'ntsc': 'TRUE', 'timebase': '29.97'}, 'timecode': {'displayformat': 'NDF', 'rate': {'ntsc': 'TRUE', 'timebase': '29.97'}}, 'uuid': '377abc11-205d-498b-b058-6e73c151d97f'}}



# # Further data editing 
# 
# ## Create a camera view column that matches the audio_video_tuple to  idxmax

# In[7]:

def XMLgenerator():
    data_df=pd.read_pickle('idxmax.pkl')


    # In[8]:


    #use indexes of when crossover is 1 to change cam_view to 0 for three seconds after crossover


    # In[9]:


    cap = cv2. VideoCapture(VIDEO_FILEPATH_LIST[0])
    vlength = int(cap. get(cv2. CAP_PROP_FRAME_COUNT))
    alength=len(data_df)


    # In[10]:


    alength
    vlength


    # In[11]:


    vrate=24
    arate=500
    arate_actual=24


    # ## Create intervals of data in tuple form. 
    # 
    # ## idxmax_mic_data are Video Tuples which  represent the start and end frame of main speaker. Can be used for audio data as well when there is sufficient mic bleed.
    # 
    # ## threshold_mic_data are Audio Tuples represent the start and end frame of when mic is being used.
    # 
    # 
    # ## Main cam tuples use the main cam column in the dataframe to change to center cam

    # In[12]:




    # In[13]:


    #create lists of tuples for each camera based on the audio

    list_of_idxmax_mic_data=[]
    for i in range(len(AUDIO_VIDEO_TUPLE_LIST)+1):
        tuple_list=dataframe_getintervals(data_df['idxmax'],i)
        list_of_idxmax_mic_data.append(tuple_list)


    ##create lists of tuples for each Audio


    # In[14]:


    #Audio Tuples
    #creating a list to referencing the column names of df 
    list_str_audio_thresholds=[]
    for irow in range(len(AUDIO_VIDEO_TUPLE_LIST)):
        varname="A"
        list_str_audio_thresholds.append(varname+str(irow))


    # In[15]:


    #create lists of tuples for each audio based the threshold region
    list_of_tuples_threshold_mic_data=[]
    for i in range(len(AUDIO_VIDEO_TUPLE_LIST)):
        tuple_list=dataframe_getintervals(data_df[list_str_audio_thresholds[i]],1)
        list_of_tuples_threshold_mic_data.append(tuple_list)


    # In[16]:


    # main cam tuple list
    main_cam_tuple_list=dataframe_getintervals(data_df['Main_cam'],1)


    # In[17]:


    main_cam_tuple_list_corrected=[]
    guard=0
    count=0
    #add 3 seconds to the end of every transition
    for i in range(len(main_cam_tuple_list)):

        if guard==0:

            if i+1==len(main_cam_tuple_list):
                pass
            else:
                if main_cam_tuple_list[i+1][0]-main_cam_tuple_list[i][1]<arate*3:
                    #create tuple
                    new_tuple=(main_cam_tuple_list[i][0],main_cam_tuple_list[i+1][1])
                    #add tuple to list
                    main_cam_tuple_list_corrected.append(new_tuple)
                    guard=1

                else:
                    main_cam_tuple_list_corrected.append(main_cam_tuple_list[i])

        else:
            guard=0


    # In[18]:


    extended_audio_idxmax_mic_data=[]

    for i in range(len(list_of_idxmax_mic_data)):
    #for each list of tuple in the list
        new_tuple_list=[]

        for i2 in range(len(list_of_idxmax_mic_data[i])):
        #for each element in tuple, add 1.5 seconds to the clip

            extended_tuple=(list_of_idxmax_mic_data[i][i2][0]+arate*1.5,list_of_idxmax_mic_data[i][i2][1]+arate*1.5)



            new_tuple_list.append(extended_tuple)



        extended_audio_idxmax_mic_data.append(new_tuple_list)


    # ## Create extra center cam intervals for long unchanged camera focus

    # In[19]:


    #append bits of center cam to diversify long lengths of non-center cam
    for lists_of_tuples in list_of_idxmax_mic_data:
        for tuples in lists_of_tuples:
            length=tuples[1]-tuples[0]
            if tuples[1]-tuples[0]>30*arate:
                interval_num=math.floor(length/(15*arate))
                if tuples[1]<tuples[0]+interval_num*15*arate+10*arate:
                    interval_num=interval_num-1
                    print(True)


                    for i in range(interval_num):

                        tup_start=tuples[0]+(i+1)*15*arate
                        tup_end=tup_start+8*arate
                        new_tuple=(tup_start,tup_end)
                        print(new_tuple , tuples ,length/arate)
                        main_cam_tuple_list_corrected.append(new_tuple)

                #interval=tuple()
            #main_cam_tuple_list.append


    # In[20]:


    main_cam_tuple_list_corrected.sort()


    # In[21]:


    #skeletal outline of otio


    # build the structure
    tl = otio.schema.Timeline(name="Example timeline", metadata=example_metadata)

    # add track for each video file and each audio file
    #for each file add a track

    #create lists for each track to reference back to later
    #vtr is video track, etc.
    vtr_list=[]
    atr_list=[]


    #add a audio AND video track for each video track
        #default cam first because it is lowest priority
    vtr_default = otio.schema.Track(name="Default_camera", kind="Video")
    tl.tracks.append(vtr_default)

    for i in AUDIO_VIDEO_TUPLE_LIST:
        atr = otio.schema.Track(name=i[0], kind="Audio")
        tl.tracks.append(atr)
        atr_list.append(atr)

        #video
        vtr = otio.schema.Track(name=i[0]+"_video", kind="Video")
        tl.tracks.append(vtr)
        vtr_list.append(vtr)

    # add main cam
    main_tr = otio.schema.Track(name='Main_Cam')
    tl.tracks.append(main_tr)


    # In[22]:


    vrate=24
    arate=500
    arate_actual=24


    # # Two different loops, One for Video, another for audio

    # In[23]:


    #video clips
    # i is to keep track which audio file we are currently on
    i=-1

    # i2 is to keep track of how many clips there are
    i2=0

    for lists_of_tuples in list_of_idxmax_mic_data:
        #connect the list of tuples to the audio file
        #ignore 0 for now. We will come back to this later, i will still be -1 but we will add a default clip instead

        if i==-1:
        #add the entire default cam clip in the lowest priority
            #vclip
            #aclip
            pass

        else: 
            #print(AUDIO_VIDEO_TUPLE_LIST[i],VIDEO_FILEPATH_LIST[AUDIO_VIDEO_TUPLE_LIST[i][1]])

            # Connect the audio and video tracks 
            atrack=atr_list[i]
            afname=AUDIO_VIDEO_TUPLE_LIST[i][0]

            # AUDIO_VIDEO_TUPLE_LIST[i][1] references the VIDEO_FILEPATH_LIST to determine which video file to use
            vtrack=vtr_list[i]
            vfname=VIDEO_FILEPATH_LIST[AUDIO_VIDEO_TUPLE_LIST[i][1]]



            #will remember the previous end of clip's timecode
            #will reset to 0 when a video-audio track pair is done
            vprevious_end_timecode=0
            aprevious_end_timecode=0

        #=========================================================================

            for tuples in lists_of_tuples:

            #=====================================================
              #adding variables for time and duration calculation
                #Video clips 
                v_clip_starttime=tuples[0]/arate*vrate
                v_clip_duration=tuples[1]/arate*vrate-v_clip_starttime


            #Specifying the start time and end time of the video file where clips come from
                v_clip_available_range=otio.opentime.TimeRange(
                start_time=otio.opentime.RationalTime(0, vrate),
                duration=otio.opentime.RationalTime(vlength, vrate))

                vref = otio.schema.ExternalReference(target_url=vfname,
                available_range=v_clip_available_range)


            #specifying where the start timecode and end timecode of the clip is
                v_clip_source_range=otio.opentime.TimeRange(
                    start_time=otio.opentime.RationalTime(v_clip_starttime, vrate),
                    duration=otio.opentime.RationalTime(v_clip_duration, vrate))



            #Video Gaps
                #the gap length is equal to (timecode of next clip time - timecode end of previous clip)
                v_gap_start_time=0
                v_gap_duration=v_clip_starttime-vprevious_end_timecode

                v_gap_timerange=otio.opentime.TimeRange(
                    start_time=otio.opentime.RationalTime(v_gap_start_time, vrate),
                    duration=otio.opentime.RationalTime(v_gap_duration, vrate))




             #=====================================================   
                # adding gaps before adding clips
                #convert the tuple ranges from audio rate to video rate
                    #audio rate=500
                    #video rate=24



                # create gap settings
                vgap = otio.schema.Gap(
                    name="vGap{}".format(i2 + 1),

                    # available_range_from_list is the 
                    source_range=otio.opentime.TimeRange(
                        start_time=otio.opentime.RationalTime(
                            v_gap_timerange.start_time.value,
                            v_gap_timerange.start_time.rate
                        ),
                        duration=otio.opentime.RationalTime(
                            v_gap_timerange.duration.value,
                            v_gap_timerange.duration.rate
                        ),
                    )
                )

                # put the clip into the track
                vtrack.append(vgap)



            #=======================================================
            # adding Video clips 
            #convert the tuple ranges from audio rate to video rate
            #audio rate=500

                            #add clip to track                

                vcl = otio.schema.Clip(
                            name="vClip{}".format(i2 + 1),
                            media_reference=vref,

                            # available_range_from_list is the 
                            source_range=otio.opentime.TimeRange(
                                start_time=otio.opentime.RationalTime(
                                    v_clip_source_range.start_time.value,
                                    v_clip_source_range.start_time.rate
                                ),
                                duration=otio.opentime.RationalTime(
                                    v_clip_source_range.duration.value,
                                    v_clip_source_range.duration.rate
                                ),
                            )
                        )

                vtrack.append(vcl)

                vprevious_end_timecode=tuples[1]/arate*vrate
                #apreviousduration=



                i2=i2+1


        i=i+1
        #if i == 2:
            #break


    # # Loudness Threshold Audio method

    # In[25]:


    # i is to keep track which audio file we are currently on
    i=0

    # i2 is to keep track of how many clips there are
    i2=0

    for lists_of_tuples in list_of_tuples_threshold_mic_data:
        #connect the list of tuples to the audio file
        #ignore 0 for now. We will come back to this later, i will still be -1 but we will add a default clip instead

        if i==-1:
        #add the entire default cam clip in the lowest priority
            #vclip
            #aclip
            pass

        else: 
            print(AUDIO_VIDEO_TUPLE_LIST[i],VIDEO_FILEPATH_LIST[AUDIO_VIDEO_TUPLE_LIST[i][1]])

            # Connect the audio and video tracks 
            atrack=atr_list[i]
            afname=AUDIO_VIDEO_TUPLE_LIST[i][0]

            # AUDIO_VIDEO_TUPLE_LIST[i][1] references the VIDEO_FILEPATH_LIST to determine which video file to use
            vtrack=vtr_list[i]
            vfname=VIDEO_FILEPATH_LIST[AUDIO_VIDEO_TUPLE_LIST[i][1]]



            #will remember the previous end of clip's timecode
            #will reset to 0 when a video-audio track pair is done
            vprevious_end_timecode=0
            aprevious_end_timecode=0

        #=========================================================================

            for tuples in lists_of_tuples:

            #=====================================================
              #adding variables for time and duration calculation
                #Video clips 
                a_clip_starttime=tuples[0]/arate*arate_actual
                a_clip_duration=tuples[1]/arate*arate_actual-a_clip_starttime


            #Specifying the start time and end time of the video file where clips come from
                a_clip_available_range=otio.opentime.TimeRange(
                start_time=otio.opentime.RationalTime(0, arate_actual),
                duration=otio.opentime.RationalTime(alength*arate_actual, arate_actual))

                aref = otio.schema.ExternalReference(target_url=afname,
                available_range=a_clip_available_range)


            #specifying where the start timecode and end timecode of the clip is
                a_clip_source_range=otio.opentime.TimeRange(
                    start_time=otio.opentime.RationalTime(a_clip_starttime, arate_actual),
                    duration=otio.opentime.RationalTime(a_clip_duration, arate_actual))



            #Video Gaps
                #the gap length is equal to (timecode of next clip time - timecode end of previous clip)
                a_gap_start_time=0
                a_gap_duration=a_clip_starttime-aprevious_end_timecode

                a_gap_timerange=otio.opentime.TimeRange(
                    start_time=otio.opentime.RationalTime(a_gap_start_time, arate_actual),
                    duration=otio.opentime.RationalTime(a_gap_duration, arate_actual))




             #=====================================================   
                # adding gaps before adding clips
                #convert the tuple ranges from audio rate to video rate
                    #audio rate=500
                    #video rate=24



                # create gap settings
                agap = otio.schema.Gap(
                    name="vGap{}".format(i2 + 1),

                    # available_range_from_list is the 
                    source_range=otio.opentime.TimeRange(
                        start_time=otio.opentime.RationalTime(
                            a_gap_timerange.start_time.value,
                            a_gap_timerange.start_time.rate
                        ),
                        duration=otio.opentime.RationalTime(
                            a_gap_timerange.duration.value,
                            a_gap_timerange.duration.rate
                        ),
                    )
                )

                # put the clip into the track
                atrack.append(agap)



            #=======================================================
            # adding Video clips 
            #convert the tuple ranges from audio rate to video rate
            #audio rate=500

                            #add clip to track                

                acl = otio.schema.Clip(
                            name="aClip{}".format(i2 + 1),
                            media_reference=aref,

                            # available_range_from_list is the 
                            source_range=otio.opentime.TimeRange(
                                start_time=otio.opentime.RationalTime(
                                    a_clip_source_range.start_time.value,
                                    a_clip_source_range.start_time.rate
                                ),
                                duration=otio.opentime.RationalTime(
                                    a_clip_source_range.duration.value,
                                    a_clip_source_range.duration.rate
                                ),
                            )
                        )

                atrack.append(acl)

                aprevious_end_timecode=tuples[1]/arate*arate_actual
                #apreviousduration=



                i2=i2+1


        i=i+1
        #if i == 2:
            #break


    # # Main cam column
    # 

    # In[26]:


    vtrack=main_tr
    vfname=VIDEO_FILEPATH_LIST[0]

    #will remember the previous end of clip's timecode
    #will reset to 0 when a video-audio track pair is done
    vprevious_end_timecode=0
    aprevious_end_timecode=0

    #=========================================================================
    i=0
    i2=0
    for tuples in main_cam_tuple_list_corrected:

    #=====================================================
      #adding variables for time and duration calculation
        #Video clips 
        v_clip_starttime=tuples[0]/arate*vrate
        v_clip_duration=tuples[1]/arate*vrate-v_clip_starttime


    #Specifying the start time and end time of the video file where clips come from
        v_clip_available_range=otio.opentime.TimeRange(
        start_time=otio.opentime.RationalTime(0, vrate),
        duration=otio.opentime.RationalTime(vlength, vrate))

        vref = otio.schema.ExternalReference(target_url=vfname,
        available_range=v_clip_available_range)


    #specifying where the start timecode and end timecode of the clip is
        v_clip_source_range=otio.opentime.TimeRange(
            start_time=otio.opentime.RationalTime(v_clip_starttime, vrate),
            duration=otio.opentime.RationalTime(v_clip_duration, vrate))



    #Video Gaps
        #the gap length is equal to (timecode of next clip time - timecode end of previous clip)
        v_gap_start_time=0
        v_gap_duration=v_clip_starttime-vprevious_end_timecode

        v_gap_timerange=otio.opentime.TimeRange(
            start_time=otio.opentime.RationalTime(v_gap_start_time, vrate),
            duration=otio.opentime.RationalTime(v_gap_duration, vrate))




     #=====================================================   
        # adding gaps before adding clips
        #convert the tuple ranges from audio rate to video rate
            #audio rate=500
            #video rate=24



        # create gap settings
        vgap = otio.schema.Gap(
            name="vGap{}".format(i2 + 1),

            # available_range_from_list is the 
            source_range=otio.opentime.TimeRange(
                start_time=otio.opentime.RationalTime(
                    v_gap_timerange.start_time.value,
                    v_gap_timerange.start_time.rate
                ),
                duration=otio.opentime.RationalTime(
                    v_gap_timerange.duration.value,
                    v_gap_timerange.duration.rate
                ),
            )
        )

        # put the clip into the track
        vtrack.append(vgap)



    #=======================================================
    # adding Video clips 
    #convert the tuple ranges from audio rate to video rate
    #audio rate=500

                    #add clip to track                

        vcl = otio.schema.Clip(
                    name="vClip{}".format(i2 + 1),
                    media_reference=vref,

                    # available_range_from_list is the 
                    source_range=otio.opentime.TimeRange(
                        start_time=otio.opentime.RationalTime(
                            v_clip_source_range.start_time.value,
                            v_clip_source_range.start_time.rate
                        ),
                        duration=otio.opentime.RationalTime(
                            v_clip_source_range.duration.value,
                            v_clip_source_range.duration.rate
                        ),
                    )
                )

        vtrack.append(vcl)

        vprevious_end_timecode=tuples[1]/arate*vrate
        #apreviousduration=



        i2=i2+1


    i=i+1
    #if i == 2:
    #break


    # In[27]:


    #otio.adapters.write_to_file(tl, 'kden_output.kdenlive')
    otio.adapters.write_to_file(tl, 'xml_output.xml')


    # In[28]:


    #otio.adapters.write_to_file(tl, 'videoaudio_beta2_main.kdenlive')
    #otio.adapters.write_to_file(tl, 'videoaudio_beta2_main.xml')


    # # The clip adding via tuple looping

    # In[ ]:




