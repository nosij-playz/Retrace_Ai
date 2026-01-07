from psp.corverge import AgeConnector 
    connector = AgeConnector(
        predictor_path="shape_predictor_68_face_landmarks.dat"
    )

    connector.connect(
        deaged_path="/content/lal_as_child.png",
        aged_path="/content/final_head_enhanced.jpg",
        target_age=25,
        output_path="/content/final_connected_face.jpg"
    )
