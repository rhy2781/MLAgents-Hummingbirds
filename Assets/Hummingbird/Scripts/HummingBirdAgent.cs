using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

/// <summary>
/// A humming bird machine learning agent
/// </summary> 
public class HummingBirdAgent : Agent
{
    [Tooltip("Force to apply when moving")]
    public float moveForce = 2f;

    [Tooltip("Speed to pitch up or down")]
    public float pitchSpeed = 100f;

    [Tooltip("Speed to rotate around the up axis")]
    public float yawSpeed = 100f;

    [Tooltip("Transform at the tip of the beak")]
    public Transform beakTip;

    [Tooltip("Whether this is training mode or Gameplay mode")]
    public bool trainingMode;

    // The rigid body of the agent
    public Rigidbody rigidBody;

    // The flower area that the agent is in 
    private FlowerArea flowerArea;

    // The nearest flower to the agent
    private Flower nearestFlower;

    // Allows for smoother pitch changes
    private float smoothPitchChange = 0f;

    // Allows for smoother yaw changes
    private float smoothYawChange = 0f;

    // Maximum angle that the bird can pitch up or down
    private const float maxPitchAngle = 80f;

    // Maximum distance from the beak tip to accept nectar collision
    private const float beakTipRadius = 0.008f;

    // Whether the agent is frozen(intentionally not fying)
    // observing the environment, but not taking any actions
    private bool frozen = false;

    // The current distance from the nearest hunter
    private HunterAgent nearestHunterAgent;
    private HunterAgent nextHunterAgent;
    private float distanceToHunter;

    /// <summary>
    /// The amount of nectar this agent has obtained this episode
    /// </summary> x
    public float NectarObtained
    {
        get;
        private set;
    }

    /// <summary>
    /// Initialize the agent
    /// </summary> 
    public override void Initialize()
    {
        rigidBody = GetComponent<Rigidbody>();
        flowerArea = GetComponentInParent<FlowerArea>();
    }


    /// <summary>
    /// Called when an action is received from either the player or the neural network
    ///
    /// vectorAction[i] represents:
    /// Index 0: move vector x (+1 right, -1 left)
    /// Index 1: move vector y (+1 up, -1 down)
    /// Index 2: move vector z (+1 forward, -1 backward)
    /// Index 3: pitch angle (+1 pitch up, -1 pitch down)
    /// Index 4: yaw angle (+1 turn right, -1 turn left)
    /// <paran name="actionBuffers">The actions to take</param>
    /// </summary> 

    public override void OnActionReceived(ActionBuffers vectorAction)
    {
        // Don't take actions if frozen
        if (frozen) return;

        var continuousActions = vectorAction.ContinuousActions;

        // calculate movement vector
        Vector3 move = new(continuousActions[0], continuousActions[1], continuousActions[2]);

        // add force in the direction of the move vector
        rigidBody.AddForce(move * moveForce);

        // Get the current rotation
        Vector3 rotationVector = transform.rotation.eulerAngles;

        // calculate pitch and yaw rotation 
        float pitchChange = continuousActions[3];
        float yawChange = continuousActions[4];

        // calculate mooth rotation changes
        smoothPitchChange = Mathf.MoveTowards(smoothPitchChange, pitchChange, 2f * Time.fixedDeltaTime);
        smoothYawChange = Mathf.MoveTowards(smoothYawChange, yawChange, 2f * Time.fixedDeltaTime);

        // calculate new pitch and yaw based on new smoothed values
        // clamp pitch to avoid flipping upside down
        float pitch = rotationVector.x + smoothPitchChange * Time.fixedDeltaTime * pitchSpeed;
        if (pitch > 180) pitch -= 360f;
        pitch = Mathf.Clamp(pitch, -maxPitchAngle, maxPitchAngle);

        float yaw = rotationVector.y + smoothYawChange * Time.fixedDeltaTime * yawSpeed;

        // apply the new rotation
        transform.rotation = Quaternion.Euler(pitch, yaw, 0f);
    }

    /// <summary>
    /// Collect vector observations from the environment
    /// <param name="sensor">The vector sensor</param>
    /// </summary> 
    public override void CollectObservations(VectorSensor sensor)
    {
        // If nearest flower is null, observe an empty array and return early
        if (nearestFlower == null || nearestHunterAgent == null)
        {
            sensor.AddObservation(new float[11]);
            return;
        }

        // Observe the agent's local rotation (4 observations)
        sensor.AddObservation(transform.localRotation.normalized); // Quaternion with magitude of 1

        // Get a vector from the beak tip to the nearest flower
        Vector3 toFlower = nearestFlower.FlowerCenterPosition - beakTip.position;

        // Observe a normalized vector pointing to the nearest flower(3 observations)
        sensor.AddObservation(toFlower.normalized);

        // Observe a dot product that indicates whether the beak tip is in front of the flower(1 observation)
        // (+1 means that the beak tip is directly in front of the flower, -1 means directly behind)
        sensor.AddObservation(Vector3.Dot(toFlower.normalized, -nearestFlower.FlowerUpVector.normalized)); // beak to flower and down vector into the flower

        // Observe a dot product that indicates whether the beak is pointing towards the flower(1 observation)
        // (+1 means that the beak is pointing directly at the flower, -1 means directly away)
        sensor.AddObservation(Vector3.Dot(beakTip.forward.normalized, -nearestFlower.FlowerUpVector.normalized)); // orientation of the beak to the flower and down vector into the flower

        // Observe the relative distance from the beak tip to the flower(1 observation)
        sensor.AddObservation(toFlower.magnitude / FlowerArea.AreaDiameter); // ratio to the flower compared to total island diameter

        Vector3 toNearestHunter = nearestHunterAgent.rigidBody.position - rigidBody.position;
        sensor.AddObservation(toNearestHunter.magnitude / FlowerArea.AreaDiameter); // ratio of the distance to hunter to total island diameter

        // 11 total observations
    }

    /// <summary>
    /// When behavior type is set to hueristic only on the agent's behavior parameters,
    /// this function will be called. It's return values will be fed into
    /// <see cref="OnActionReceived(float[])">Instead of using the neural network</cref>
    /// <param name="actionsOut">An output action array</param>
    /// </summary>
    
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        /** Currently disabled for debugging purposes
        ActionSegment<float> continuousActions = actionsOut.ContinuousActions;

        // Create placeholders for all movement/turning
        Vector3 forward = Vector3.zero;
        Vector3 left = Vector3.zero;
        Vector3 up = Vector3.zero;
        float pitch = 0f;
        float yaw = 0f;

        // Forward - Backward
        if (Input.GetKey(KeyCode.W)) forward = transform.forward;
        else if (Input.GetKey(KeyCode.S)) forward = -transform.forward;

        // Left - Right
        if (Input.GetKey(KeyCode.A)) left = -transform.right;
        else if (Input.GetKey(KeyCode.D)) left = transform.right;

        // Up - Down
        if (Input.GetKey(KeyCode.E)) up = transform.up;
        else if (Input.GetKey(KeyCode.C)) up = -transform.up;

        // Pitch up - Pitch down
        if (Input.GetKey(KeyCode.UpArrow)) pitch = -0.5f;
        else if (Input.GetKey(KeyCode.DownArrow)) pitch = 0.5f;

        // Turn left - Turn right
        if (Input.GetKey(KeyCode.LeftArrow)) yaw = -0.5f;
        else if (Input.GetKey(KeyCode.RightArrow)) yaw = 0.5f;

        // Combin the movement vectors and normalize
        Vector3 combined = (forward + left + up).normalized;

        continuousActions[0] = combined.x;
        continuousActions[1] = combined.y;
        continuousActions[2] = combined.z;
        continuousActions[3] = pitch;
        continuousActions[4] = yaw;
        **/
    }

    /// <summary>
    /// Pevent the agent from moving and taking actions
    /// </summary>
    public void FreezeAgent()
    {
        Debug.Assert(trainingMode == false, "Freeze/Unfreeze not supported in training");
        frozen = true;
        rigidBody.Sleep();
    }

    /// <summary>
    /// Resume Agent movement and actions
    /// </summary>
    public void UnfreezeAgent()
    {
        Debug.Assert(trainingMode == false, "Freeze/Unfreeze not supported in training");
        frozen = false;
        rigidBody.WakeUp();
    }

    /// <summary>
    /// Move the agent into a safe random posotion i.e. does not collide anything 
    /// If also in front of flower, point the beak in front of the flower
    /// </summary> 
    private void MoveToSafeRandomPosition()
    {
        bool safePositionFound = false;
        int attemptsRemaining = 100; // Prevents infinite loop
        Vector3 potentialPosition = Vector3.zero;
        Quaternion potentialRotation = new();

        // loop until a safe position is found, or we run out of attempts
        while(!safePositionFound && attemptsRemaining > 0)
        {
            attemptsRemaining --;
            // Pick a random flower
            Flower randomFlower = flowerArea.Flowers[Random.Range(0, flowerArea.Flowers.Count)];

            // Position 10 cm in front of the flower
            potentialPosition = randomFlower.transform.position + randomFlower.FlowerUpVector * .1f;

            // Point beak at flower(bird's head is center of transform)
            Vector3 toFlower = randomFlower.FlowerCenterPosition - potentialPosition;
            potentialRotation = Quaternion.LookRotation(toFlower, Vector3.up); 
           

            // Check to see if the agent will collide with anything
            Collider[] colliders = Physics.OverlapSphere(potentialPosition, 0.05f);
        
            // Sage position has been found is no colliders have overlapped
            safePositionFound = colliders.Length == 0;
        }
        Debug.Assert(safePositionFound, "Could not find a safe position to spawn");
        transform.SetPositionAndRotation(potentialPosition, potentialRotation);
    }

    /// <summary>
    /// Update the nearest flower to the agent
    /// </summary> 
    private void UpdateNearestFlower()
    {
        // Choose the first flower in the area and iterate to see if there is a closer flower
        Flower nextFlower = flowerArea.Flowers[0];
        foreach(Flower potentialFlower in flowerArea.Flowers)
        {
            if (potentialFlower.HasNectar)
            {
                float distanceToNextFlower = Vector3.Distance(nextFlower.transform.position, beakTip.position);
                float distanceToPotentialFlower = Vector3.Distance(potentialFlower.transform.position, beakTip.position);

                if(distanceToPotentialFlower < distanceToNextFlower)
                {
                    nextFlower = potentialFlower;
                }
            }
        }
        nearestFlower = nextFlower;
    }

    /// <summary>
    /// Update the nearest Hunter to the humming bird in case there is multiple hunters
    /// </summary>
    private void UpdateNearestHunter()
    {
        // if we only have one hunter in the area
        if(flowerArea.Hunters.Count == 1)
        {
            nearestHunterAgent = flowerArea.Hunters[0];
            distanceToHunter = Vector3.Distance(nearestHunterAgent.rigidBody.position, rigidBody.position);
            return;
        }

        // if we have multiple hunters in the area
        if(nextHunterAgent == null)
        {
            nextHunterAgent = flowerArea.Hunters[Random.Range(0, flowerArea.Hunters.Count)];
        }
        foreach(HunterAgent potentialHunterAgent in flowerArea.Hunters)
        {
            float distanceToNextHunter = Vector3.Distance(nextHunterAgent.rigidBody.position, transform.position);
            float distanceToPotentialHunter = Vector3.Distance(potentialHunterAgent.rigidBody.position, transform.position);

            if(distanceToPotentialHunter < distanceToNextHunter)
            {
                nextHunterAgent = potentialHunterAgent;
            }
        }
        nearestHunterAgent = nextHunterAgent;
        distanceToHunter = Vector3.Distance(nearestHunterAgent.rigidBody.position, rigidBody.position);
    }

    /// <summary>
    /// Called when the agent's collider enters a trigger collider
    /// </summary>
    /// <param name="other">The trigger collider</param>
    private void OnTriggerEnter(Collider other)
    {
        TriggerEnterOrStay(other);
    }

    /// <summary>
    /// Called when the agent's collider stays in a trigger collider
    /// </summary>
    /// <param name="other">The trigger collider</param>
    private void OnTriggerStay(Collider other)
    {
        TriggerEnterOrStay(other);
    }

    /// <summary>
    /// Handles when the agent's collider enters or stays in a trigger collider
    /// </summary>
    /// <param name="collider">The trigger collider</param>
    private void TriggerEnterOrStay(Collider collider)
    {
        // Check if agent is colliding with nectar
        if(collider.CompareTag("nectar"))
        {
            Vector3 closestPointToBeakTip = collider.ClosestPoint(beakTip.position);


            // Check if the closest collision point is close to the beak tip
            // Note: A collision with anything but the beak tip should not count
            if (Vector3.Distance(beakTip.position, closestPointToBeakTip) < beakTipRadius)
            {
                // Look up the flower for this nectar collider
                Flower flower = flowerArea.GetFlowerFromNectar(collider);

                // Attempt to take .01 nectar
                // Note: this is per fixed timestep, meaning it happens every .02 seconds, or 50x per second
                float nectarReceived = flower.Feed(.01f);

                // Keep track of nectar obtained
                NectarObtained += nectarReceived;

                // Add rewards for the group
                float bonus = Mathf.Clamp01(Vector3.Dot(transform.forward.normalized, -nearestFlower.FlowerUpVector.normalized));
                flowerArea.mAgentGroupBird.AddGroupReward(0.01f + bonus);

                // If flower is empty, update nearest flower
                if (!flower.HasNectar)
                {
                    UpdateNearestFlower();
                }
            }
        }
    }

    /// <summary>
    /// Called when the agent collides with something solid
    /// </summary>
    /// <param name="collision">The collision info</param>
    private void OnCollisionEnter(Collision collision)
    {
        if(collision.collider.CompareTag("boundary"))
        {
            // want ot encourage the bird to not hit any borders
            flowerArea.mAgentGroupBird.AddGroupReward(-0.05f);
        }
        else if(collision.gameObject.CompareTag("hunter_agent"))
        {
            // hunter attacked the humming bird, and we want to encourage the bird to avoid the hunter
            flowerArea.mAgentGroupBird.AddGroupReward(-0.05f);
        }
    }

    /// <summary>
    /// Called every frame. This methods helps provide us with debug information about what the hummingbirds knows
    /// </summary>
    private void Update()
    {
        // Draw a line from the beak tip to the nearest flower
        if(nearestFlower != null)
        {
            Debug.DrawLine(beakTip.position, nearestFlower.FlowerCenterPosition, Color.red);
        }
        Debug.DrawLine(rigidBody.position, nearestHunterAgent.rigidBody.position, Color.yellow);
    }

    /// <summary>
    /// Called every .02 seconds
    /// This method is used for anything that applies to the rigidbody movement, so when the agent takes action in response to the
    /// locations of the hunter andnearest flower
    /// </summary>
    private void FixedUpdate()
    {
        UpdateNearestHunter();

        if (nearestFlower == null || !nearestFlower.HasNectar)
        {
            // avoids nearest scenario where nearest flower nectar is stolen by opponent and not updated
            UpdateNearestFlower();
        }
    }

    /// <summary>
    /// This method is used to make sure that all the birds that were set to inactive are active again on the begining of a new episode
    /// </summary>
    public void ResetBird()
    {
        gameObject.SetActive(true);
        NectarObtained = 0;

        rigidBody.velocity = Vector3.zero;
        rigidBody.angularVelocity = Vector3.zero;

        // move bird and update nearest flower and hunter so that the bird is aware of its new location
        MoveToSafeRandomPosition();
        UpdateNearestFlower();
        UpdateNearestHunter();
    }
}
