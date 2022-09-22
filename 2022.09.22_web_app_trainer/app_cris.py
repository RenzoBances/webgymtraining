import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import time
from PIL import Image
import base64
from random import randrange
import pandas as pd

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

st.set_page_config(
    page_title="STARTER TRAINING -UPC",
    page_icon ="img/upc_logo.png",
)

@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

img_upc=get_base64_of_bin_file('img/upc_logo_50x50.png')
st.markdown(
    f"""
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {{
        width: 336px;        
    }}
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {{
        width: 336px;
        margin-left: -336px;
        background-color: #00F;
    }}
    [data-testid="stVerticalBlock"] {{
        flex: 0;
        gap: 0;
    }}
    #starter-training{{
        padding: 0;
    }}
    #div-upc{{
        #border: 1px solid #DDDDDD;
        background-image: url("data:image/png;base64,{img_upc}");
        position: fixed !important;
        right: 14px;
        bottom: 14px;
        width: 50px;
        height: 50px;
        background-size: 50px 50px;
    }}
    .css-10trblm{{
        color: #FFF;
        font-size: 40px;
        font-family: 'PROGRESS PERSONAL USE';
        src: url(fonts/ProgressPersonalUse-EaJdz.ttf);       
    }}
    #.main {{
        background: linear-gradient(135deg,#a8e73d,#09e7db,#092de7);
        background-size: 180% 180%;
        animation: gradient-animation 3s ease infinite;
        }}

        @keyframes gradient-animation {{
            0% {{ background-position: 0% 50%; }}
            50% {{ background-position: 100% 50%; }}
            100% {{ background-position: 0% 50%; }}
        }}
    .block-container{{
        max-width: 100%;
    }}
    </style>
    """ ,
    unsafe_allow_html=True,
)

st.title('STARTER TRAINING')
st.sidebar.markdown('---')
st.markdown("<div id='div-upc'></span>", unsafe_allow_html=True)
st.sidebar.title('The Training App')
st.sidebar.markdown('---')

app_mode = st.sidebar.selectbox('Choose your training:',
    ['(home)','Glute Bridge','Abs', 'Lunges', 'Push Up', 'Squats']
)

if app_mode =='(home)':
    a=0
elif app_mode =='Glute Bridge':
    st.markdown('### __ABS__')
    st.markdown("<hr/>", unsafe_allow_html=True)

    webcam = st.checkbox('Start Webcam')
    st.markdown("Camera status: "+str(webcam))

    trainer, user = st.columns(2)

    with trainer:        
        st.markdown("TRAINER", unsafe_allow_html=True)
        experto = randrange(3)+1

        video_trainer_file="videos_trainer/Glute Bridges/Glute Bridges"+str(experto)+".mp4"
        df_experto = pd.read_csv("videos_trainer/Glute Bridges/Puntos_Glute_Brigdes"+str(experto)+".csv")
        df_costos = pd.read_csv("videos_trainer/Glute Bridges/Costos_Glute Bridge_promedio.csv")

        st.table(df_experto)

        st.video(video_trainer_file, format="video/mp4", start_time=0)
        st.table(df_costos)
        

    with user:
        st.markdown("YOU", unsafe_allow_html=True)

        def calcular_costos(ej_usuario, df_planchas_experto1, inicio, fin):
    
            #inicio=0
            #fin = 1
            
            resultados = []
            resultados_index = []
            resultados_costo_al = []
            resultados_costo_al_normalizado = []

            #while (inicio <= len(df_planchas_experto1)-1 and fin <= len(df_planchas_experto1)):
                #print()
            ej_experto = []
            for i in range(0,len(df_planchas_experto1.columns)):
                #print(i)
                ej_experto.append(df_planchas_experto1[inicio:fin][i][inicio])

            x = np.array(ej_usuario) 

            y = np.array(ej_experto)

                #plt.figure(figsize=(6, 4))
                #plt.plot(np.arange(x.shape[0]), x + 1.5, "-o", c="C3")
                #plt.plot(np.arange(y.shape[0]), y - 1.5, "-o", c="C0")
                #plt.axis("off")
                #plt.savefig("signals_a_b.pdf")

                # Distance matrix
            N = x.shape[0]
            M = y.shape[0]
            dist_mat = np.zeros((N, M))
            for i in range(N):
                for j in range(M):
                    dist_mat[i, j] = abs(x[i] - y[j])

                # DTW
            path, cost_mat = dp(dist_mat)
                #print("Alignment cost: {:.4f}".format(cost_mat[N - 1, M - 1]))
                #print("Normalized alignment cost: {:.4f}".format(cost_mat[N - 1, M - 1]/(N + M)))

                #plt.figure(figsize=(6, 4))
                #plt.subplot(121)
                #plt.title("Distance matrix")
                #plt.imshow(dist_mat, cmap=plt.cm.binary, interpolation="nearest", origin="lower")
                #plt.subplot(122)
                #plt.title("Cost matrix")
                #plt.imshow(cost_mat, cmap=plt.cm.binary, interpolation="nearest", origin="lower")
            x_path, y_path = zip(*path)
                #plt.plot(y_path, x_path);
                #[indice, aligment_cost, normalized_alignment_cost:]
            resultados_index.append(inicio)
            resultados_costo_al.append(cost_mat[N - 1, M - 1])
            resultados_costo_al_normalizado.append(cost_mat[N - 1, M - 1]/(N + M))
            resultados.append([inicio,cost_mat[N - 1, M - 1],cost_mat[N - 1, M - 1]/(N + M)])
            inicio=inicio+1
            fin = fin+1
            return resultados



        if webcam:
            #stframe = st.empty()
            #vid = cv2.VideoCapture(0)

            #width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
            #height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
            #fps_input = int(vid.get(cv2.CAP_PROP_FPS))

            #codec = cv2.VideoWriter_fourcc('V','P','0','9')
            #fps = 0
            #i = 0
            #drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=2)

            #max_faces=1
            #detection_confidence=90
            #tracking_confidence=90
            #with mp_face_mesh.FaceMesh(
            #    min_detection_confidence=detection_confidence,
            #    min_tracking_confidence=tracking_confidence, 
            #    max_num_faces = max_faces) as face_mesh:
            #    prevTime = 0

            #    while vid.isOpened():
            #        i +=1
            #        ret, frame = vid.read()
            #        if not ret:
            #            continue

            #        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #        results = face_mesh.process(frame)

            #        frame.flags.writeable = True
            #        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            #        face_count = 0
            #        if results.multi_face_landmarks:
            #            for face_landmarks in results.multi_face_landmarks:
            #                face_count += 1
            #                mp_drawing.draw_landmarks(
            #                image = frame,
            #                landmark_list=face_landmarks,
            #                connections=mp_face_mesh.FACE_CONNECTIONS,
            #                landmark_drawing_spec=drawing_spec,
            #                connection_drawing_spec=drawing_spec)
            #        currTime = time.time()
            #        fps = 1 / (currTime - prevTime)
            #        prevTime = currTime

            #        frame = cv2.resize(frame,(0,0),fx = 0.8 , fy = 0.8)
            #        stframe.image(frame,channels = 'BGR',use_column_width=True)
            #stframe = st.empty()
            #vid.release()

            vid = cv2.VideoCapture(0,cv2.CAP_DSHOW)
            mp_pose = mp.solutions.pose
            start = time.time()
            #"the code you want to test stays here"

            c=0
            val = 0
            with mp_pose.Pose(static_image_mode=False) as pose:

                while True:
                    ret, frame = vid.read()

                    if ret == False:
                        break

                    frame = cv2.flip(frame,1)
                    height, width, _ = frame.shape
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = pose.process(frame_rgb)
                    #print("Pose landmarks: ", results.pose_landmarks)
                    
                    if c % 15 == 0:
                    
                        resultados = []

                        for i in range(0, len(results.pose_landmarks.landmark)):
                            #time.sleep(1)
                            resultados.append(results.pose_landmarks.landmark[i].x)
                            resultados.append(results.pose_landmarks.landmark[i].y)
                            resultados.append(results.pose_landmarks.landmark[i].z)
                            resultados.append(results.pose_landmarks.landmark[i].visibility)

                        df_puntos = pd.DataFrame(np.reshape(resultados, (132, 1)).T)
                        df_puntos['segundo'] = str(c/15)
                        #print("Dataframe: ", df_puntos)
                        if c==0:
                            df_puntos_final = df_puntos.copy()
                        else:
                            #print(type(df_puntos_final))
                            #print(type(df_puntos))
                            df_puntos_final = pd.concat([df_puntos_final, df_puntos])
                        
                        ej_usuario = resultados
                        
                        if val == 0:
                            
                            inicio=0
                            fin = 1
                            print("Intentando el segundo cero")
                            resultados_costos = calcular_costos(ej_usuario,df_experto,inicio,fin)
                        
                            if (resultados_costos[0][1]<= df_costos.Costo_alineamiento[inicio]-df_costos.Desviacion_estandar[inicio] or resultados_costos[0][1]<= df_costos.Costo_alineamiento[inicio]+df_costos.Desviacion_estandar[inicio]) and val==0: # promedio +- desviación estandar (para evitar casos rápidos o lentos)
                            
                                val = 1
                                print("Costo desde: ", str(df_costos.Costo_alineamiento[inicio]-df_costos.Desviacion_estandar[inicio]))
                                print("Costo hasta: ", str(df_costos.Costo_alineamiento[inicio]+df_costos.Desviacion_estandar[inicio]))
                                print("Costo resultante: ", str(resultados_costos[0][1]))
                                print("Se realizó la pose del segundo cero")
                                
                            inicio = inicio+1
                            fin = fin+1    
                        else:
                            # SE VA A COMPARAR LOS COSTOS DE ALINEAMIENTO POR SEGUNDO EXACTO DEL USUARIO CON RESPECTO AL EXPERTO
                            
                            #inicio=1
                            #fin = 2
                            print("Intentando el segundo", str(inicio))
                            resultados_costos = calcular_costos(ej_usuario,df_experto,inicio,fin)
                            
                            if (resultados_costos[0][1]>= df_costos.Costo_alineamiento[inicio]-df_costos.Desviacion_estandar[inicio] and resultados_costos[0][1]<= df_costos.Costo_alineamiento[inicio]+df_costos.Desviacion_estandar[inicio]):
                                print("Costo desde: ", str(df_costos.Costo_alineamiento[inicio]-df_costos.Desviacion_estandar[inicio]))
                                print("Costo hasta: ", str(df_costos.Costo_alineamiento[inicio]+df_costos.Desviacion_estandar[inicio]))
                                print("Costo resultante: ", str(resultados_costos[0][1]))
                                print("Se realizó la pose del segundo correspondiente")
                            else:
                                print("Costo desde: ", str(df_costos.Costo_alineamiento[inicio]-df_costos.Desviacion_estandar[inicio]))
                                print("Costo hasta: ", str(df_costos.Costo_alineamiento[inicio]+df_costos.Desviacion_estandar[inicio]))
                                print("Costo resultante: ", str(resultados_costos[0][1]))
                                print("No se realizó el ejercicio correctamente!!!!!")
                            
                            inicio = inicio+1
                            fin = fin+1  
                            
                            if inicio == len(df_experto):
                                
                                inicio=0
                                fin = 1
                                val=0
                        c = c+1
                    else:
                        #print("")

                        if results.pose_landmarks is not None:

                            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                                mp_drawing.DrawingSpec(color=(128,0,250), thickness=2, circle_radius=3),
                                                mp_drawing.DrawingSpec(color=(255,255,255), thickness=2))
                        titulo_windows = "Prueba de ejercicios" 
                        cv2.imshow(titulo_windows,frame)

                        #if cv2.waitKey(1) == ord('q'):
                        #   break

                        #if (cv2.waitKey(1) | 0xFF == 27) | cv2.waitKey(10) & 0xFF == ord('q'):
                            #break

                        #if cv2.waitKey(10) & 0xFF == ord('q'):
                        #   break

                        if cv2.waitKey(1) & 0xFF == 27:
                            break
                        c = c+1

                    #key = cv2.waitKey(25)
                    #if key == ord('n') or key == ord('p') or key == ord('q'):
                    #    break

            vid.release()
            cv2.destroyAllWindows()
            end = time.time()
            print("Segundos transacurridos: ",end - start)




        else:
            vid = cv2.VideoCapture(0)
            vid.release()
        

    st.markdown("<hr/>", unsafe_allow_html=True)


elif app_mode =='Abs':
    a=0
elif app_mode =='Lunges':
    a=0
elif app_mode =='Push Up':
    a=0
elif app_mode =='Squats':
    a=0
else:
    a=0

