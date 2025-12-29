package com.pilvae.engine

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.pilvae.engine.ui.theme.PILVAETheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            PILVAETheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    MainScreen(
                        onRunSimulation = {
                            startActivity(
                                android.content.Intent(this, SimulationActivity::class.java)
                            )
                        }
                    )
                }
            }
        }
    }
}

@Composable
fun MainScreen(onRunSimulation: () -> Unit) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Text(
            text = "PIL-VAE Engine",
            fontSize = 32.sp,
            fontWeight = FontWeight.Bold,
            color = MaterialTheme.colorScheme.primary
        )

        Spacer(modifier = Modifier.height(8.dp))

        Text(
            text = "Gradient-Free Learning System",
            fontSize = 16.sp,
            color = MaterialTheme.colorScheme.secondary
        )

        Spacer(modifier = Modifier.height(48.dp))

        Card(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.surfaceVariant
            )
        ) {
            Column(
                modifier = Modifier.padding(16.dp)
            ) {
                Text(
                    text = "Features",
                    fontWeight = FontWeight.SemiBold,
                    fontSize = 18.sp
                )
                Spacer(modifier = Modifier.height(8.dp))
                FeatureItem("✓ Bi-Directional PIL Layers")
                FeatureItem("✓ Ridge Regression Solver")
                FeatureItem("✓ Sherman-Morrison Updates")
                FeatureItem("✓ Numerical Stability Monitor")
                FeatureItem("✓ VAE Architecture")
            }
        }

        Spacer(modifier = Modifier.height(32.dp))

        Button(
            onClick = onRunSimulation,
            modifier = Modifier
                .fillMaxWidth()
                .height(56.dp)
                .padding(horizontal = 16.dp)
        ) {
            Text(
                text = "Run Simulation",
                fontSize = 18.sp
            )
        }

        Spacer(modifier = Modifier.height(16.dp))

        Text(
            text = "No Backpropagation • Matrix Inversion Only",
            fontSize = 12.sp,
            color = MaterialTheme.colorScheme.outline
        )
    }
}

@Composable
fun FeatureItem(text: String) {
    Text(
        text = text,
        fontSize = 14.sp,
        modifier = Modifier.padding(vertical = 4.dp)
    )
}
